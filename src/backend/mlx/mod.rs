use crate::arch::{H, HD, NLAYER, NH, weight_names};
use crate::engine::{BackendId, DType, Engine, EngineInfo, ModelSpec, Out, Req};
use crate::graph::{self, Weights};
use crate::ops::{Backend, WeightFormat};
use crate::systemone::Calibration;
use crate::weights::Tensors;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Device, Dtype, Stream, fast, ops};
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Debug)]
enum MlxData {
    Dense(Array),
    Quant {
        wq: Array,
        scales: Array,
        biases: Array,
        group: i32,
        bits: i32,
    },
}

#[derive(Clone, Debug)]
pub struct MlxTensor {
    data: MlxData,
    shape: Vec<usize>,
}

impl MlxTensor {
    fn dense(a: Array) -> MlxTensor {
        let shape = a.shape().iter().map(|x| *x as usize).collect();
        MlxTensor {
            data: MlxData::Dense(a),
            shape,
        }
    }

    fn quant(
        wq: Array,
        scales: Array,
        biases: Array,
        group: i32,
        bits: i32,
        shape: Vec<usize>,
    ) -> MlxTensor {
        MlxTensor {
            data: MlxData::Quant {
                wq,
                scales,
                biases,
                group,
                bits,
            },
            shape,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn array(&self) -> &Array {
        match &self.data {
            MlxData::Dense(a) => a,
            MlxData::Quant { .. } => panic!("mlx: expected a dense tensor"),
        }
    }
}

fn scalar(v: f32) -> Array {
    Array::from_f32(v)
}

fn gelu(x: &Array) -> Array {
    let e = ops::erf(&x.multiply(&scalar(0.7071067811865476)).expect("gelu mul"))
        .expect("gelu erf");
    x.multiply(&e.add(&scalar(1.0)).expect("gelu add"))
        .expect("gelu mul")
        .multiply(&scalar(0.5))
        .expect("gelu mul")
}

fn rot_half(x: &Array, cs: &Array, sn: &Array, half: usize) -> Array {
    let parts = x
        .split_at_indices(&[half as i32], Some(-1))
        .expect("rope split");
    let x1 = &parts[0];
    let x2 = &parts[1];
    let lo = x1
        .multiply(cs)
        .expect("rope mul")
        .subtract(&x2.multiply(sn).expect("rope mul"))
        .expect("rope sub");
    let hi = x2
        .multiply(cs)
        .expect("rope mul")
        .add(&x1.multiply(sn).expect("rope mul"))
        .expect("rope add");
    ops::concatenate(&[lo, hi], -1).expect("rope concat")
}

fn eligible(dims: &[usize], group: i32) -> bool {
    dims.len() == 2 && group > 0 && dims[1] % group as usize == 0
}

fn take_rows(t: &MlxTensor, ia: &Array) -> Array {
    match &t.data {
        MlxData::Dense(a) => a.take_axis(ia, 0).expect("take rows"),
        MlxData::Quant {
            wq,
            scales,
            biases,
            group,
            bits,
        } => {
            let w = wq.take_axis(ia, 0).expect("take rows wq");
            let s = scales.take_axis(ia, 0).expect("take rows scales");
            let b = biases.take_axis(ia, 0).expect("take rows biases");
            ops::dequantize(&w, &s, Some(&b), *group, *bits).expect("take rows dequantize")
        }
    }
}

pub struct MlxBackend {
    quant: Option<(i32, i32)>,
    cache: Option<HashMap<String, (Array, Array, Array)>>,
}

impl MlxBackend {
    pub fn dense() -> MlxBackend {
        MlxBackend {
            quant: None,
            cache: None,
        }
    }

    fn quantized(bits: i32, group: i32, cache: Option<HashMap<String, (Array, Array, Array)>>) -> MlxBackend {
        MlxBackend {
            quant: Some((bits, group)),
            cache,
        }
    }
}

impl Backend for MlxBackend {
    type Tensor = MlxTensor;

    fn name(&self) -> &'static str {
        "mlx"
    }

    fn sync(&self) {
        let rc = unsafe { mlx_sys::mlx_synchronize(Stream::thread_local_or_default().as_ptr()) };
        assert_eq!(rc, 0, "mlx_synchronize failed with status {}", rc);
    }

    fn upload_f32(&self, data: &[f32], shape: &[usize]) -> MlxTensor {
        let sh: Vec<i32> = shape.iter().map(|x| *x as i32).collect();
        MlxTensor::dense(Array::from_slice(data, &sh))
    }

    fn download_f32(&self, t: &MlxTensor) -> Vec<f32> {
        let c = match &t.data {
            MlxData::Dense(a) => a.contiguous().expect("mlx contiguous"),
            MlxData::Quant {
                wq,
                scales,
                biases,
                group,
                bits,
            } => ops::dequantize(wq, scales, Some(biases), *group, *bits)
                .expect("mlx dequantize")
                .contiguous()
                .expect("mlx contiguous"),
        };
        c.eval().expect("mlx eval");
        c.try_as_slice::<f32>()
            .expect("mlx as_slice f32")
            .to_vec()
    }

    fn weight_format(&self, _name: &str, dims: &[usize]) -> WeightFormat {
        let Some((bits, group)) = self.quant else {
            return WeightFormat::F32;
        };
        if !eligible(dims, group) {
            return WeightFormat::F32;
        }
        WeightFormat::Quant { bits, group }
    }

    fn resident(&self, name: &str) -> bool {
        self.cache.as_ref().is_some_and(|c| c.contains_key(name))
    }

    fn upload_quant(
        &self,
        name: &str,
        data: &[f32],
        shape: &[usize],
        bits: i32,
        group: i32,
    ) -> MlxTensor {
        if let Some((wq, scales, biases)) = self.cache.as_ref().and_then(|c| c.get(name)) {
            return MlxTensor::quant(
                wq.clone(),
                scales.clone(),
                biases.clone(),
                group,
                bits,
                shape.to_vec(),
            );
        }
        let sh: Vec<i32> = shape.iter().map(|x| *x as i32).collect();
        let w = Array::from_slice(data, &sh);
        let (wq, scales, biases) = ops::quantize(&w, group, bits).expect("mlx quantize");
        MlxTensor::quant(wq, scales, biases, group, bits, shape.to_vec())
    }

    fn embedding(&self, table: &MlxTensor, ids: &[i64]) -> MlxTensor {
        let idx: Vec<i32> = ids.iter().map(|x| *x as i32).collect();
        let ia = Array::from_slice(&idx, &[idx.len() as i32]);
        MlxTensor::dense(take_rows(table, &ia))
    }

    fn layernorm(
        &self,
        x: &MlxTensor,
        w: &MlxTensor,
        b: Option<&MlxTensor>,
        eps: f32,
    ) -> MlxTensor {
        let bias = b.map(|t| t.array());
        MlxTensor::dense(
            fast::layer_norm(&x.array(), w.array(), bias, eps).expect("mlx layer_norm"),
        )
    }

    fn matmul(&self, x: &MlxTensor, w: &MlxTensor) -> MlxTensor {
        let out = match &w.data {
            MlxData::Dense(a) => x.array().matmul(a).expect("mlx matmul"),
            MlxData::Quant {
                wq,
                scales,
                biases,
                group,
                bits,
            } => ops::quantized_matmul(x.array(), wq, scales, Some(biases), false, *group, *bits)
                .expect("mlx quantized_matmul"),
        };
        MlxTensor::dense(out)
    }

    fn matmul_t(&self, x: &MlxTensor, w: &MlxTensor) -> MlxTensor {
        let out = match &w.data {
            MlxData::Dense(a) => {
                let wt = a.transpose().expect("mlx transpose");
                x.array().matmul(&wt).expect("mlx matmul_t")
            }
            MlxData::Quant {
                wq,
                scales,
                biases,
                group,
                bits,
            } => ops::quantized_matmul(x.array(), wq, scales, Some(biases), true, *group, *bits)
                .expect("mlx quantized_matmul_t"),
        };
        MlxTensor::dense(out)
    }

    fn split_qkv_rope(
        &self,
        qkv: &MlxTensor,
        rope: Option<(&MlxTensor, &MlxTensor)>,
        scale: f32,
        nh: usize,
        hd: usize,
        len: usize,
    ) -> (MlxTensor, MlxTensor, MlxTensor) {
        let n = qkv.shape[0] / len;
        let half = hd / 2;
        let x = qkv
            .array()
            .reshape(&[n as i32, len as i32, 3, nh as i32, hd as i32])
            .expect("qkv reshape");
        let parts = x.split_at_indices(&[1, 2], Some(2)).expect("qkv split");
        let q = parts[0].squeeze_axes(&[2]).expect("q squeeze");
        let k = parts[1].squeeze_axes(&[2]).expect("k squeeze");
        let v = parts[2].squeeze_axes(&[2]).expect("v squeeze");
        let (q, k) = match rope {
            Some((cos, sin)) => {
                let cs = cos
                    .array()
                    .reshape(&[1, len as i32, 1, half as i32])
                    .expect("cos reshape");
                let sn = sin
                    .array()
                    .reshape(&[1, len as i32, 1, half as i32])
                    .expect("sin reshape");
                (
                    rot_half(&q, &cs, &sn, half),
                    rot_half(&k, &cs, &sn, half),
                )
            }
            None => (q, k),
        };
        let s = scalar(scale);
        let q = q
            .multiply(&s)
            .expect("q scale")
            .swap_axes(1, 2)
            .expect("q swap");
        let k = k
            .multiply(&s)
            .expect("k scale")
            .swap_axes(1, 2)
            .expect("k swap");
        let v = v.swap_axes(1, 2).expect("v swap");
        (
            MlxTensor::dense(q),
            MlxTensor::dense(k),
            MlxTensor::dense(v),
        )
    }

    fn attention(
        &self,
        q: &MlxTensor,
        k: &MlxTensor,
        v: &MlxTensor,
        mask: &MlxTensor,
        nh: usize,
        len: usize,
    ) -> MlxTensor {
        let mask4 = mask.array().expand_dims(1).expect("mask expand");
        let out = fast::scaled_dot_product_attention(
            &q.array(),
            &k.array(),
            &v.array(),
            1.0,
            &mask4,
            None::<&Array>,
        )
        .expect("mlx sdpa");
        let merged_heads = nh * out.shape()[3] as usize;
        let rows = out.shape()[0] as usize * len;
        let merged = out
            .transpose_axes(&[0, 2, 1, 3])
            .expect("attn transpose")
            .reshape(&[rows as i32, merged_heads as i32])
            .expect("attn reshape");
        MlxTensor::dense(merged)
    }

    fn window_mask(&self, att: &[i64], n: usize, len: usize, window: Option<usize>) -> MlxTensor {
        let a: Vec<i32> = att.iter().map(|x| *x as i32).collect();
        let key = Array::from_slice(&a, &[n as i32, len as i32]);
        let keep = key
            .ne(Array::from_int(0))
            .expect("mask ne")
            .expand_dims(-2)
            .expect("mask expand");
        let mut drop = keep.logical_not().expect("mask not");
        if let Some(w) = window {
            let idx = ops::arange::<i32, i32>(None::<i32>, len as i32, None::<i32>)
                .expect("mask arange");
            let row = idx.reshape(&[len as i32, 1]).expect("mask reshape");
            let col = idx.reshape(&[1, len as i32]).expect("mask reshape");
            let d = row
                .subtract(&col)
                .expect("mask sub")
                .abs()
                .expect("mask abs");
            let far = ops::gt(&d, &Array::from_int(w as i32))
                .expect("mask gt")
                .expand_dims(0)
                .expect("mask expand");
            drop = drop.logical_or(&far).expect("mask or");
        }
        let drop = ops::broadcast_to(&drop, &[n as i32, len as i32, len as i32])
            .expect("mask broadcast");
        MlxTensor::dense(
            ops::select(&drop, &scalar(-3.4e38), &scalar(0.0)).expect("mask select"),
        )
    }

    fn add(&self, a: &MlxTensor, b: &MlxTensor) -> MlxTensor {
        MlxTensor::dense(a.array().add(b.array()).expect("mlx add"))
    }

    fn add_bias(&self, x: &MlxTensor, b: &MlxTensor) -> MlxTensor {
        MlxTensor::dense(x.array().add(b.array()).expect("mlx add_bias"))
    }

    fn gelu(&self, x: &MlxTensor) -> MlxTensor {
        MlxTensor::dense(gelu(&x.array()))
    }

    fn gelu_mul(&self, x: &MlxTensor, half: usize) -> MlxTensor {
        let rows = x.shape[0];
        let r = x
            .array()
            .reshape(&[rows as i32, 2, half as i32])
            .expect("gelu_mul reshape");
        let parts = r.split_at_indices(&[1], Some(1)).expect("gelu_mul split");
        let g = gelu(&parts[0]);
        let out = g.multiply(&parts[1]).expect("gelu_mul mul");
        MlxTensor::dense(
            out.reshape(&[rows as i32, half as i32])
                .expect("gelu_mul reshape"),
        )
    }

    fn relu(&self, x: &MlxTensor) -> MlxTensor {
        MlxTensor::dense(ops::maximum(&x.array(), &scalar(0.0)).expect("mlx relu"))
    }

    fn add_type(
        &self,
        x: &MlxTensor,
        type_emb: &MlxTensor,
        qtype: &[i64],
        len: usize,
    ) -> MlxTensor {
        let rows = x.shape[0];
        let idx: Vec<i32> = (0..rows).map(|r| qtype[r / len] as i32).collect();
        let ia = Array::from_slice(&idx, &[rows as i32]);
        let te = take_rows(type_emb, &ia);
        MlxTensor::dense(x.array().add(&te).expect("add_type add"))
    }

    fn gather_markers(&self, x: &MlxTensor, mpos: &[i64], k: usize) -> MlxTensor {
        let len = x.shape[0] / (mpos.len() / k);
        let idx: Vec<i32> = (0..mpos.len())
            .map(|r| ((r / k) * len + mpos[r] as usize) as i32)
            .collect();
        let ia = Array::from_slice(&idx, &[idx.len() as i32]);
        MlxTensor::dense(x.array().take_axis(&ia, 0).expect("gather take"))
    }

    fn post(&self, sc: &MlxTensor, mmask: &[i64], k: usize) -> (MlxTensor, MlxTensor) {
        let n = sc.shape[0] / k;
        let mm: Vec<i32> = mmask.iter().map(|x| *x as i32).collect();
        let m = Array::from_slice(&mm, &[n as i32, k as i32]);
        let s = sc
            .array()
            .reshape(&[n as i32, k as i32])
            .expect("post reshape");
        let vis = m.ne(Array::from_int(0)).expect("post ne");
        let masked = ops::select(&vis, &s, &scalar(-10000.0)).expect("post select");
        let logits = masked
            .reshape(&[(n * k) as i32])
            .expect("post reshape logits");

        let p = ops::softmax_axis(&masked, -1, Some(true)).expect("post softmax");
        let p = p.contiguous().expect("post contiguous");
        let sorted = ops::sort_axis(&p, -1).expect("post sort");
        let t1 = sorted
            .try_index((.., (k - 1) as i32..k as i32))
            .expect("post index t1")
            .squeeze_axes(&[-1])
            .expect("post squeeze t1");
        let t2 = if k >= 2 {
            sorted
                .try_index((.., (k - 2) as i32..(k - 1) as i32))
                .expect("post index t2")
                .squeeze_axes(&[-1])
                .expect("post squeeze t2")
        } else {
            ops::zeros::<f32>(&[n as i32]).expect("post zeros")
        };
        let t2 = ops::maximum(&t2, &scalar(0.0)).expect("post clamp t2");
        let second = t1.subtract(&t2).expect("post second");

        let pc = ops::maximum(&p, &scalar(1e-9)).expect("post clamp p");
        let lp = pc.log().expect("post log");
        let ent = p
            .multiply(&lp)
            .expect("post ent mul")
            .sum_axis(-1, false)
            .expect("post ent sum")
            .negative()
            .expect("post ent neg");

        let nv = vis
            .as_dtype(Dtype::Float32)
            .expect("post nv cast")
            .sum_axis(-1, false)
            .expect("post nv sum");
        let nn = ops::maximum(&nv, &scalar(2.0)).expect("post nn max");
        let third = ent
            .divide(&nn.log().expect("post nn log"))
            .expect("post third");
        let fourth = nn.divide(&scalar(255.0)).expect("post fourth");

        let feats = ops::stack(&[t1, second, third, fourth], -1).expect("post stack");
        (MlxTensor::dense(logits), MlxTensor::dense(feats))
    }

    fn act_input(&self, x: &MlxTensor, feats: &MlxTensor, len: usize) -> MlxTensor {
        let dim = x.shape[1];
        let n = feats.shape[0];
        let r = x
            .array()
            .reshape(&[n as i32, len as i32, dim as i32])
            .expect("act reshape");
        let first = r.try_index((.., 0, ..)).expect("act index");
        MlxTensor::dense(
            ops::concatenate(&[first, feats.array().clone()], -1).expect("act concat"),
        )
    }
}

pub struct ConvertStats {
    pub names: usize,
    pub quantized: usize,
    pub params: usize,
    pub quant_params: usize,
    pub bytes: u64,
}

pub fn convert(
    gguf: &str,
    _alt: Option<&str>,
    _manifest: Option<&str>,
    bits: i32,
    group: i32,
    out: &str,
) -> ConvertStats {
    assert!(
        group == 32 || group == 64 || group == 128,
        "mlx affine group must be 32, 64 or 128, got {}",
        group
    );
    assert!((2..=8).contains(&bits), "mlx affine bits must be 2..8, got {}", bits);
    let t = Tensors::open(gguf);
    let names = weight_names();
    let mut arrays: Vec<(String, Array)> = Vec::new();
    let mut stats = ConvertStats {
        names: names.len(),
        quantized: 0,
        params: 0,
        quant_params: 0,
        bytes: 0,
    };
    for n in &names {
        let (v, dims) = t.f32_shaped(n);
        let params: usize = dims.iter().product();
        stats.params += params;
        if !eligible(&dims, group) {
            continue;
        }
        let sh: Vec<i32> = dims.iter().map(|x| *x as i32).collect();
        let w = Array::from_slice(&v, &sh);
        let (wq, scales, biases) = ops::quantize(&w, group, bits).expect("mlx quantize");
        stats.quantized += 1;
        stats.quant_params += params;
        arrays.push((format!("{}.weight", n), wq));
        arrays.push((format!("{}.scales", n), scales));
        arrays.push((format!("{}.biases", n), biases));
    }
    for (_, a) in &arrays {
        a.eval().expect("mlx eval");
    }
    let kept: Vec<(String, Array)> = arrays
        .iter()
        .map(|(n, a)| (n.clone(), a.contiguous().expect("mlx contiguous")))
        .collect();
    let views: Vec<(String, TensorView)> = kept
        .iter()
        .map(|(n, a)| (n.clone(), TensorView::try_from(a).expect("safetensors view")))
        .collect();
    let mut meta = HashMap::new();
    meta.insert("format".to_string(), "mlx-affine".to_string());
    meta.insert("bits".to_string(), bits.to_string());
    meta.insert("group".to_string(), group.to_string());
    if let Some(p) = Path::new(out).parent() {
        if !p.as_os_str().is_empty() {
            std::fs::create_dir_all(p).expect("convert: cannot create output directory");
        }
    }
    safetensors::serialize_to_file(views, Some(meta), Path::new(out)).expect("convert: write");
    stats.bytes = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);
    stats
}

fn read_quantized(path: &str) -> (i32, i32, HashMap<String, (Array, Array, Array)>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {}", path, e));
    let st = SafeTensors::deserialize(&bytes).unwrap_or_else(|e| panic!("{}: {:?}", path, e));
    let (_, header) =
        SafeTensors::read_metadata(&bytes).unwrap_or_else(|e| panic!("{}: {:?}", path, e));
    let meta = header.metadata().clone().unwrap_or_default();
    let get = |k: &str, d: i32| -> i32 {
        meta.get(k).and_then(|v| v.parse().ok()).unwrap_or(d)
    };
    let bits = get("bits", 4);
    let group = get("group", 64);
    let mut map = HashMap::new();
    for name in weight_names() {
        let wk = format!("{}.weight", name);
        let sk = format!("{}.scales", name);
        let bk = format!("{}.biases", name);
        let (Ok(w), Ok(s), Ok(b)) = (st.tensor(&wk), st.tensor(&sk), st.tensor(&bk)) else {
            continue;
        };
        map.insert(
            name,
            (
                Array::try_from(w).expect("safetensors wq"),
                Array::try_from(s).expect("safetensors scales"),
                Array::try_from(b).expect("safetensors biases"),
            ),
        );
    }
    (bits, group, map)
}

pub struct MlxEngine {
    bk: MlxBackend,
    w: Weights<MlxBackend>,
    dtype: DType,
    cal: Calibration,
}

impl MlxEngine {
    pub fn load(spec: &ModelSpec) -> MlxEngine {
        let quant_path = Path::new(&spec.base)
            .parent()
            .map(|p| p.join("mlx/weights.safetensors").to_string_lossy().into_owned());
        let file_source = spec.quantized
            && quant_path
                .as_ref()
                .is_some_and(|p| Path::new(p).exists());
        let mut bits = spec.quant_bits;
        let mut group = spec.quant_group;
        let mut cache = None;
        if file_source {
            let path = quant_path.clone().unwrap();
            let (b, g, c) = read_quantized(&path);
            eprintln!(
                "mlx: {} quantized tensors from {} (bits {}, group {})",
                c.len(),
                path,
                b,
                g
            );
            bits = b;
            group = g;
            cache = Some(c);
        }
        assert!(
            group == 32 || group == 64 || group == 128,
            "mlx affine group must be 32, 64 or 128, got {}",
            group
        );
        let bk = if file_source {
            MlxBackend::quantized(bits, group, cache)
        } else {
            MlxBackend::dense()
        };
        let tensors = Tensors::open(&spec.base);
        let cal = tensors.meta.calibration();
        let dtype = tensors.meta.dtype();
        let w = graph::load(&bk, &tensors);
        drop(tensors);
        MlxEngine {
            bk,
            w,
            dtype,
            cal,
        }
    }
}

impl Engine for MlxEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            id: BackendId::Mlx,
            device: format!(
                "{} {}",
                self.bk.name(),
                Device::try_default().expect("mlx default device")
            ),
            dtype: self.dtype,
            layers: NLAYER,
            calibration: self.cal.clone(),
        }
    }

    fn forward(&mut self, req: &Req) -> Out {
        graph::forward(&self.bk, &self.w, req, None)
    }
}

const _: () = assert!(H == NH * HD);
