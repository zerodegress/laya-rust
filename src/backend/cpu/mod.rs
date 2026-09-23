use crate::arch::{H, HD, NLAYER, NH};
use crate::engine::{BackendId, DType, Engine, EngineInfo, ModelSpec, Out, Req};
use crate::graph::{self, Weights};
use crate::ops::Backend;
use crate::systemone::Calibration;
use crate::weights::Tensors;

#[derive(Clone, Debug)]
pub struct CpuTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl CpuTensor {
    fn new(data: Vec<f32>, shape: Vec<usize>) -> CpuTensor {
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "shape {:?} does not match {} elements",
            shape,
            data.len()
        );
        CpuTensor { data, shape }
    }

    fn rows(&self) -> usize {
        self.shape[0]
    }

    fn cols(&self) -> usize {
        self.shape[1]
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dim(&self) -> usize {
        *self.shape.last().unwrap()
    }
}

fn nthreads() -> usize {
    std::env::var("LAYA_CPU_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|v| v.get())
                .unwrap_or(1)
        })
}

fn erf(x: f32) -> f32 {
    let x = x as f64;
    let p = 0.3275911f64;
    let a = [
        0.254829592f64,
        -0.284496736f64,
        1.421413741f64,
        -1.453152027f64,
        1.061405429f64,
    ];
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let poly = t * (a[0] + t * (a[1] + t * (a[2] + t * (a[3] + t * a[4]))));
    (sign * (1.0 - poly * (-x * x).exp())) as f32
}

fn gelu1(v: f32) -> f32 {
    0.5 * v * (1.0 + erf(v / 1.4142135381698608))
}

fn gemm_nt(x: &[f32], w: &[f32], r: usize, kk: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0f32; r * n];
    let threads = nthreads().min(r.max(1));
    let chunk = r.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (ti, yblock) in y.chunks_mut(chunk * n).enumerate() {
            let row0 = ti * chunk;
            s.spawn(move || {
                for (ri, yrow) in yblock.chunks_mut(n).enumerate() {
                    let xr = &x[(row0 + ri) * kk..(row0 + ri) * kk + kk];
                    for (kx, &a) in xr.iter().enumerate() {
                        let wr = &w[kx * n..kx * n + n];
                        for ni in 0..n {
                            yrow[ni] += a * wr[ni];
                        }
                    }
                }
            });
        }
    });
    y
}

fn gemm_t(x: &[f32], w: &[f32], r: usize, kk: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0f32; r * n];
    let threads = nthreads().min(r.max(1));
    let chunk = r.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (ti, yblock) in y.chunks_mut(chunk * n).enumerate() {
            let row0 = ti * chunk;
            s.spawn(move || {
                for (ri, yrow) in yblock.chunks_mut(n).enumerate() {
                    let xr = &x[(row0 + ri) * kk..(row0 + ri) * kk + kk];
                    for (ni, slot) in yrow.iter_mut().enumerate() {
                        let wr = &w[ni * kk..ni * kk + kk];
                        let mut acc = 0f32;
                        for kx in 0..kk {
                            acc += xr[kx] * wr[kx];
                        }
                        *slot = acc;
                    }
                }
            });
        }
    });
    y
}

pub struct CpuBackend;

impl Backend for CpuBackend {
    type Tensor = CpuTensor;

    fn name(&self) -> &'static str {
        "cpu"
    }

    fn upload_f32(&self, data: &[f32], shape: &[usize]) -> CpuTensor {
        CpuTensor::new(data.to_vec(), shape.to_vec())
    }

    fn download_f32(&self, t: &CpuTensor) -> Vec<f32> {
        t.data.clone()
    }

    fn embedding(&self, table: &CpuTensor, ids: &[i64]) -> CpuTensor {
        let dim = table.dim();
        let mut out = vec![0f32; ids.len() * dim];
        for (r, id) in ids.iter().enumerate() {
            let src = *id as usize * dim;
            out[r * dim..r * dim + dim].copy_from_slice(&table.data[src..src + dim]);
        }
        CpuTensor::new(out, vec![ids.len(), dim])
    }

    fn layernorm(
        &self,
        x: &CpuTensor,
        w: &CpuTensor,
        b: Option<&CpuTensor>,
        eps: f32,
    ) -> CpuTensor {
        let (rows, dim) = (x.rows(), x.cols());
        let mut out = vec![0f32; rows * dim];
        for r in 0..rows {
            let xr = &x.data[r * dim..r * dim + dim];
            let or = &mut out[r * dim..r * dim + dim];
            let mean = xr.iter().map(|v| *v as f64).sum::<f64>() / dim as f64;
            let var = xr
                .iter()
                .map(|v| {
                    let d = *v as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / dim as f64;
            let rstd = 1.0 / (var + eps as f64).sqrt();
            for i in 0..dim {
                let mut v = ((xr[i] as f64 - mean) * rstd) as f32 * w.data[i];
                if let Some(b) = b {
                    v += b.data[i];
                }
                or[i] = v;
            }
        }
        CpuTensor::new(out, x.shape.clone())
    }

    fn matmul(&self, x: &CpuTensor, w: &CpuTensor) -> CpuTensor {
        let (r, kk) = (x.rows(), x.cols());
        assert_eq!(w.shape[0], kk, "matmul inner dim mismatch");
        let n = w.shape[1];
        CpuTensor::new(gemm_nt(&x.data, &w.data, r, kk, n), vec![r, n])
    }

    fn matmul_t(&self, x: &CpuTensor, w: &CpuTensor) -> CpuTensor {
        let (r, kk) = (x.rows(), x.cols());
        assert_eq!(w.shape[1], kk, "matmul_t inner dim mismatch");
        let n = w.shape[0];
        CpuTensor::new(gemm_t(&x.data, &w.data, r, kk, n), vec![r, n])
    }

    fn split_qkv_rope(
        &self,
        qkv: &CpuTensor,
        rope: Option<(&CpuTensor, &CpuTensor)>,
        scale: f32,
        nh: usize,
        hd: usize,
        len: usize,
    ) -> (CpuTensor, CpuTensor, CpuTensor) {
        let rows = qkv.rows();
        let n = rows / len;
        let half = hd / 2;
        let stride = 3 * nh * hd;
        let mut q = vec![0f32; n * nh * len * hd];
        let mut k = vec![0f32; n * nh * len * hd];
        let mut v = vec![0f32; n * nh * len * hd];
        for b in 0..n {
            for s in 0..len {
                for h in 0..nh {
                    let base = (b * len + s) * stride + h * hd;
                    for d in 0..hd {
                        let (mut qx, mut kx) = (qkv.data[base + d], qkv.data[base + nh * hd + d]);
                        if let Some((cos, sin)) = rope {
                            let j = d % half;
                            let c = cos.data[s * half + j];
                            let sn = sin.data[s * half + j];
                            let qr = if d < half {
                                -qkv.data[base + d + half]
                            } else {
                                qkv.data[base + d - half]
                            };
                            let kr = if d < half {
                                -qkv.data[base + nh * hd + d + half]
                            } else {
                                qkv.data[base + nh * hd + d - half]
                            };
                            qx = qx * c + qr * sn;
                            kx = kx * c + kr * sn;
                        }
                        let o = ((b * nh + h) * len + s) * hd + d;
                        q[o] = qx * scale;
                        k[o] = kx * scale;
                        v[o] = qkv.data[base + 2 * nh * hd + d];
                    }
                }
            }
        }
        let shape = vec![n, nh, len, hd];
        (
            CpuTensor::new(q, shape.clone()),
            CpuTensor::new(k, shape.clone()),
            CpuTensor::new(v, shape),
        )
    }

    fn attention(
        &self,
        q: &CpuTensor,
        k: &CpuTensor,
        v: &CpuTensor,
        mask: &CpuTensor,
        nh: usize,
        len: usize,
    ) -> CpuTensor {
        let n = q.shape[0];
        let hd = q.shape[3];
        let mut out = vec![0f32; n * len * nh * hd];
        let mut scores = vec![0f32; len];
        for b in 0..n {
            for h in 0..nh {
                let head = ((b * nh + h) * len) * hd;
                let qb = &q.data[head..head + len * hd];
                let kb = &k.data[head..head + len * hd];
                let vb = &v.data[head..head + len * hd];
                let mb = &mask.data[b * len * len..(b + 1) * len * len];
                for s in 0..len {
                    let qs = &qb[s * hd..s * hd + hd];
                    let ms = &mb[s * len..s * len + len];
                    let mut mx = -3.4e38f32;
                    for t in 0..len {
                        let kt = &kb[t * hd..t * hd + hd];
                        let mut acc = 0f32;
                        for d in 0..hd {
                            acc += qs[d] * kt[d];
                        }
                        let sc = acc + ms[t];
                        scores[t] = sc;
                        if sc > mx {
                            mx = sc;
                        }
                    }
                    let mut sum = 0f32;
                    for t in 0..len {
                        let e = (scores[t] - mx).exp();
                        scores[t] = e;
                        sum += e;
                    }
                    let inv = 1.0 / sum;
                    for t in 0..len {
                        scores[t] *= inv;
                    }
                    for d in 0..hd {
                        let mut acc = 0f32;
                        for t in 0..len {
                            acc += scores[t] * vb[t * hd + d];
                        }
                        out[(b * len + s) * nh * hd + h * hd + d] = acc;
                    }
                }
            }
        }
        CpuTensor::new(out, vec![n * len, nh * hd])
    }

    fn window_mask(&self, att: &[i64], n: usize, len: usize, window: Option<usize>) -> CpuTensor {
        let mut m = vec![0f32; n * len * len];
        for b in 0..n {
            for s in 0..len {
                for t in 0..len {
                    let drop = att[b * len + t] == 0
                        || window.is_some_and(|w| s.abs_diff(t) > w);
                    m[(b * len + s) * len + t] = if drop { -3.4e38 } else { 0.0 };
                }
            }
        }
        CpuTensor::new(m, vec![n, len, len])
    }

    fn add(&self, a: &CpuTensor, b: &CpuTensor) -> CpuTensor {
        assert_eq!(a.data.len(), b.data.len(), "add shape mismatch");
        let data = a
            .data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| x + y)
            .collect();
        CpuTensor::new(data, a.shape.clone())
    }

    fn add_bias(&self, x: &CpuTensor, b: &CpuTensor) -> CpuTensor {
        let dim = b.data.len();
        let mut data = x.data.clone();
        for (i, v) in data.iter_mut().enumerate() {
            *v += b.data[i % dim];
        }
        CpuTensor::new(data, x.shape.clone())
    }

    fn gelu(&self, x: &CpuTensor) -> CpuTensor {
        CpuTensor::new(x.data.iter().map(|v| gelu1(*v)).collect(), x.shape.clone())
    }

    fn gelu_mul(&self, x: &CpuTensor, half: usize) -> CpuTensor {
        let rows = x.rows();
        let stride = 2 * half;
        assert_eq!(x.data.len(), rows * stride);
        let mut out = vec![0f32; rows * half];
        for r in 0..rows {
            for c in 0..half {
                let gate = gelu1(x.data[r * stride + c]);
                out[r * half + c] = gate * x.data[r * stride + half + c];
            }
        }
        CpuTensor::new(out, vec![rows, half])
    }

    fn relu(&self, x: &CpuTensor) -> CpuTensor {
        CpuTensor::new(x.data.iter().map(|v| v.max(0.0)).collect(), x.shape.clone())
    }

    fn add_type(
        &self,
        x: &CpuTensor,
        type_emb: &CpuTensor,
        qtype: &[i64],
        len: usize,
    ) -> CpuTensor {
        let dim = x.cols();
        let mut data = x.data.clone();
        for (i, v) in data.iter_mut().enumerate() {
            let seq = (i / dim) / len;
            *v += type_emb.data[qtype[seq] as usize * dim + (i % dim)];
        }
        CpuTensor::new(data, x.shape.clone())
    }

    fn gather_markers(&self, x: &CpuTensor, mpos: &[i64], k: usize) -> CpuTensor {
        let dim = x.cols();
        let len = x.rows() / (mpos.len() / k);
        let mut out = vec![0f32; mpos.len() * dim];
        for (r, p) in mpos.iter().enumerate() {
            let b = r / k;
            let src = (b * len + *p as usize) * dim;
            out[r * dim..r * dim + dim].copy_from_slice(&x.data[src..src + dim]);
        }
        CpuTensor::new(out, vec![mpos.len(), dim])
    }

    fn post(&self, sc: &CpuTensor, mmask: &[i64], k: usize) -> (CpuTensor, CpuTensor) {
        let n = sc.data.len() / k;
        let mut logits = vec![0f32; n * k];
        let mut feats = vec![0f32; n * 4];
        for b in 0..n {
            let s = &sc.data[b * k..b * k + k];
            let m = &mmask[b * k..b * k + k];
            let mut pb = vec![0f32; k];
            for j in 0..k {
                let v = if m[j] != 0 { s[j] } else { -10000.0 };
                logits[b * k + j] = v;
                pb[j] = v;
            }
            let mx = pb.iter().cloned().fold(-3.4e38f32, f32::max);
            let mut sum = 0f32;
            for v in pb.iter_mut() {
                let e = (*v - mx).exp();
                *v = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            let mut t1 = -1f32;
            let mut t2 = -1f32;
            let mut ent = 0f32;
            let mut nv = 0usize;
            for j in 0..k {
                let p = pb[j] * inv;
                if p > t1 {
                    t2 = t1;
                    t1 = p;
                } else if p > t2 {
                    t2 = p;
                }
                ent -= p * p.max(1e-9).ln();
                if m[j] != 0 {
                    nv += 1;
                }
            }
            if t2 < 0.0 {
                t2 = 0.0;
            }
            let nn = if nv < 2 { 2 } else { nv } as f32;
            feats[b * 4] = t1;
            feats[b * 4 + 1] = t1 - t2;
            feats[b * 4 + 2] = ent / nn.ln();
            feats[b * 4 + 3] = nn / 255.0;
        }
        (
            CpuTensor::new(logits, vec![n * k]),
            CpuTensor::new(feats, vec![n, 4]),
        )
    }

    fn act_input(&self, x: &CpuTensor, feats: &CpuTensor, len: usize) -> CpuTensor {
        let dim = x.cols();
        let n = feats.shape[0];
        let nf = feats.cols();
        let mut out = vec![0f32; n * (dim + nf)];
        for b in 0..n {
            out[b * (dim + nf)..b * (dim + nf) + dim]
                .copy_from_slice(&x.data[b * len * dim..b * len * dim + dim]);
            out[b * (dim + nf) + dim..(b + 1) * (dim + nf)]
                .copy_from_slice(&feats.data[b * nf..(b + 1) * nf]);
        }
        CpuTensor::new(out, vec![n, dim + nf])
    }
}

pub struct CpuEngine {
    bk: CpuBackend,
    w: Weights<CpuBackend>,
    dtype: DType,
    cal: Calibration,
}

impl CpuEngine {
    pub fn load(spec: &ModelSpec) -> CpuEngine {
        let tensors = Tensors::open(&spec.base);
        let cal = tensors.meta.calibration();
        let dtype = tensors.meta.dtype();
        let bk = CpuBackend;
        let w = graph::load(&bk, &tensors);
        drop(tensors);
        CpuEngine { bk, w, dtype, cal }
    }
}

impl Engine for CpuEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            id: BackendId::Cpu,
            device: format!("{} x{}", self.bk.name(), nthreads()),
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
