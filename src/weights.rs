use crate::arch;
use crate::gguf::{self, Header};
use serde_json::{json, Map, Value};
use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

pub fn f2h(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e32 = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;
    if e32 == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = e32 - 112;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut r = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && (r & 1) == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) {
        r += 1;
    }
    sign | r as u16
}

pub fn h2f(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            let mut m = m;
            let mut e = -14i32;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            s | (((e + 127) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | (((e + 112) as u32) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

fn die(m: &str) -> ! {
    eprintln!("{}", m);
    std::process::exit(2);
}

pub struct ModelMeta {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub vocab: usize,
    pub scale: f32,
    pub eps: f32,
    pub window: usize,
    pub global_every: usize,
    pub head_layers: usize,
    pub act_input: usize,
    pub question_types: usize,
    pub max_len: usize,
    pub head_max_len: usize,
    pub max_prefixes: usize,
    pub file_type: u32,
    pub temperature: [f32; 3],
    pub temperature_by_options: Vec<(String, f32)>,
}

impl ModelMeta {
    pub fn defaults() -> ModelMeta {
        ModelMeta {
            hidden: arch::H,
            layers: arch::NLAYER,
            heads: arch::NH,
            head_dim: arch::HD,
            inter: arch::INTER,
            vocab: arch::NVOCAB,
            scale: arch::SCALE,
            eps: arch::EPS,
            window: arch::WINDOW as usize,
            global_every: arch::GLOBAL_EVERY,
            head_layers: arch::HEAD_LAYERS,
            act_input: arch::ACT_INPUT,
            question_types: arch::QTYPES,
            max_len: 512,
            head_max_len: 192,
            max_prefixes: 6,
            file_type: gguf::FT_MOSTLY_F16,
            temperature: [1.0, 1.0, 1.0],
            temperature_by_options: Vec::new(),
        }
    }

    pub fn dtype(&self) -> crate::engine::DType {
        match self.file_type {
            gguf::FT_ALL_F32 => crate::engine::DType::F32,
            gguf::FT_MOSTLY_F16 => crate::engine::DType::F16,
            other => die(&format!(
                "general.file_type is {}, but this build only handles {} (F32) and {} (mostly F16)",
                other,
                gguf::FT_ALL_F32,
                gguf::FT_MOSTLY_F16
            )),
        }
    }

    pub fn calibration(&self) -> crate::systemone::Calibration {
        crate::systemone::Calibration {
            temperature: self.temperature,
            by_options: self.temperature_by_options.clone(),
        }
    }

    fn from_header(h: &Header, path: &str) -> ModelMeta {
        let a = h.str_or("general.architecture", "");
        if a != "laya" {
            die(&format!(
                "{}: general.architecture is {:?}, want \"laya\"",
                path, a
            ));
        }
        let d = ModelMeta::defaults();
        let num = |key: &str, want: usize| -> usize {
            match h.get(key) {
                None => {
                    die(&format!("{}: missing metadata {}", path, key));
                }
                Some(v) => match v.as_u64() {
                    None => die(&format!("{}: metadata {} is not an integer", path, key)),
                    Some(got) if got as usize != want => die(&format!(
                        "{}: metadata {} is {} but this build pins {}",
                        path, key, got, want
                    )),
                    Some(_) => want,
                },
            }
        };
        let flt = |key: &str, want: f32| -> f32 {
            match h.get(key) {
                None => die(&format!("{}: missing metadata {}", path, key)),
                Some(v) => match v.as_f32() {
                    None => die(&format!("{}: metadata {} is not a float", path, key)),
                    Some(got) if got.to_bits() != want.to_bits() => die(&format!(
                        "{}: metadata {} is {} but this build pins {}",
                        path, key, got, want
                    )),
                    Some(_) => want,
                },
            }
        };
        let mut m = ModelMeta {
            hidden: num("laya.hidden_size", d.hidden),
            layers: num("laya.num_layers", d.layers),
            heads: num("laya.num_heads", d.heads),
            head_dim: num("laya.head_dim", d.head_dim),
            inter: num("laya.intermediate_size", d.inter),
            vocab: num("laya.vocab_size", d.vocab),
            scale: flt("laya.attention_scale", d.scale),
            eps: flt("laya.layer_norm_epsilon", d.eps),
            window: num("laya.sliding_window", d.window),
            global_every: num("laya.global_layer_every", d.global_every),
            head_layers: num("laya.head_layers", d.head_layers),
            act_input: num("laya.act_input_features", d.act_input),
            question_types: num("laya.question_types", d.question_types),
            max_len: num("laya.max_len", d.max_len),
            head_max_len: num("laya.head_max_len", d.head_max_len),
            max_prefixes: num("laya.max_prefixes", d.max_prefixes),
            file_type: h.u64_or("general.file_type", d.file_type as u64) as u32,
            ..d
        };
        for i in 0..3 {
            let key = format!("laya.temperature.{}", i);
            let v = h
                .get(&key)
                .and_then(|v| v.as_f32())
                .unwrap_or_else(|| die(&format!("{}: missing metadata {}", path, key)));
            m.temperature[i] = v;
        }
        if let Some(s) = h.get("laya.temperature_by_options").and_then(|v| v.as_str()) {
            let parsed: Value = serde_json::from_str(s)
                .unwrap_or_else(|e| die(&format!("{}: temperature_by_options: {}", path, e)));
            let o = parsed
                .as_object()
                .unwrap_or_else(|| die(&format!("{}: temperature_by_options is not an object", path)));
            let mut pairs: Vec<(String, f32)> = o
                .iter()
                .map(|(k, v)| {
                    let f = v
                        .as_f64()
                        .unwrap_or_else(|| die(&format!("{}: temperature {} is not a number", path, k)));
                    (k.clone(), f as f32)
                })
                .collect();
            pairs.sort_by(|a, b| a.0.cmp(&b.0));
            m.temperature_by_options = pairs;
        }
        m
    }
}

pub struct Tensors {
    path: String,
    header: Header,
    pub meta: ModelMeta,
    flip: HashSet<String>,
}

impl Tensors {
    pub fn open(path: &str) -> Tensors {
        let header = gguf::read(path);
        let meta = ModelMeta::from_header(&header, path);
        let flip: HashSet<String> = arch::transpose_names().into_iter().collect();
        let t = Tensors {
            path: path.to_string(),
            header,
            meta,
            flip,
        };
        t.check();
        t
    }

    fn info(&self, name: &str) -> &gguf::TensorInfo {
        self.header
            .tensor(name)
            .unwrap_or_else(|| die(&format!("{}: no tensor named {}", self.path, name)))
    }

    pub fn names(&self) -> Vec<String> {
        self.header.tensors.iter().map(|t| t.name.clone()).collect()
    }

    pub fn dtype_name(&self, name: &str) -> &'static str {
        gguf::type_name(self.info(name).ty)
    }

    pub fn stored_dims(&self, name: &str) -> Vec<usize> {
        self.info(name)
            .dims
            .iter()
            .rev()
            .map(|x| *x as usize)
            .collect()
    }

    pub fn dims(&self, name: &str) -> Vec<usize> {
        let mut d = self.stored_dims(name);
        if self.flip.contains(name) && d.len() == 2 {
            d.swap(0, 1);
        }
        d
    }

    pub fn f32(&self, name: &str) -> Vec<f32> {
        let t = self.info(name);
        let bytes = gguf::read_tensor_bytes(&self.path, &self.header, name);
        let v = gguf::decode_f32(&bytes, t.ty, name);
        if !self.flip.contains(name) {
            return v;
        }
        let d = self.stored_dims(name);
        let a = d[0];
        let c = d[1];
        let mut o = vec![0f32; v.len()];
        for i in 0..a {
            for j in 0..c {
                o[j * a + i] = v[i * c + j];
            }
        }
        o
    }

    #[cfg(any(feature = "cpu", feature = "mlx"))]
    pub fn f32_shaped(&self, name: &str) -> (Vec<f32>, Vec<usize>) {
        let d = self.dims(name);
        let v = self.f32(name);
        let prod: usize = d.iter().product();
        assert_eq!(
            prod,
            v.len(),
            "{}: shape {:?} covers {} elements but {} were read",
            name,
            d,
            prod,
            v.len()
        );
        (v, d)
    }

    pub fn dump(&self, names: &[String]) -> Value {
        let mut out = Map::new();
        for n in names {
            let t = self.info(n);
            out.insert(
                n.clone(),
                json!({
                    "dims": self.dims(n),
                    "stored_dims": self.stored_dims(n),
                    "transposed": self.flip.contains(n),
                    "type": gguf::type_name(t.ty),
                    "offset": t.offset,
                    "bytes": t.nbytes(),
                }),
            );
        }
        Value::Object(out)
    }

    fn expect(&self, name: &str, want: &[usize]) {
        let got = self.stored_dims(name);
        if got != want {
            die(&format!(
                "{}: tensor {} has stored shape {:?}, but this build pins {:?}",
                self.path, name, got, want
            ));
        }
        let ty = self.info(name).ty;
        if !matches!(ty, gguf::T_F32 | gguf::T_F16 | gguf::T_BF16) {
            die(&format!(
                "{}: tensor {} has type {} ({}), which this build does not decode",
                self.path,
                name,
                ty,
                gguf::type_name(ty)
            ));
        }
    }

    fn check(&self) {
        let h = self.meta.hidden;
        let three_h = 3 * h;
        let two_inter = 2 * self.meta.inter;
        let head_inter = arch::HEAD_INTER;
        self.expect(arch::TOK_EMB, &[self.meta.vocab, h]);
        self.expect(arch::TYPE_EMB, &[self.meta.question_types, h]);
        self.expect(arch::FINAL_NORM, &[h]);
        self.expect(arch::ROPE_FREQ_FULL, &[arch::HD / 2]);
        self.expect(arch::ROPE_FREQ_SLIDING, &[arch::HD / 2]);
        self.expect(arch::SCORER_W, &[h]);
        self.expect(arch::SCORER_B, &[h]);
        self.expect(arch::SCORER_HID, &[h, h]);
        self.expect(arch::SCORER_HID_B, &[h]);
        self.expect(arch::SCORER_OUT, &[1, h]);
        self.expect(arch::SCORER_OUT_B, &[1]);
        self.expect(arch::ACT_W1, &[arch::ACT_HIDDEN, self.meta.act_input]);
        self.expect(arch::ACT_B1, &[arch::ACT_HIDDEN]);
        self.expect(arch::ACT_W2, &[2, arch::ACT_HIDDEN]);
        self.expect(arch::ACT_B2, &[2]);
        for i in 0..self.meta.layers {
            self.expect(arch::ATTN_NORM[i], &[h]);
            self.expect(arch::MLP_NORM[i], &[h]);
            self.expect(arch::WQKV[i], &[three_h, h]);
            self.expect(arch::WO_ATTN[i], &[h, h]);
            self.expect(arch::WI[i], &[two_inter, h]);
            self.expect(arch::WO_MLP[i], &[h, self.meta.inter]);
        }
        for i in 0..self.meta.head_layers {
            self.expect(arch::H_NORM1_W[i], &[h]);
            self.expect(arch::H_NORM1_B[i], &[h]);
            self.expect(arch::H_NORM2_W[i], &[h]);
            self.expect(arch::H_NORM2_B[i], &[h]);
            self.expect(arch::H_WQKV[i], &[three_h, h]);
            self.expect(arch::H_IN_BIAS[i], &[three_h]);
            self.expect(arch::H_OUT_W[i], &[h, h]);
            self.expect(arch::H_OUT_B[i], &[h]);
            self.expect(arch::H_LIN1_W[i], &[head_inter, h]);
            self.expect(arch::H_LIN1_B[i], &[head_inter]);
            self.expect(arch::H_LIN2_W[i], &[h, head_inter]);
            self.expect(arch::H_LIN2_B[i], &[h]);
        }
    }
}

pub struct SafeTensors {
    path: String,
    header: Map<String, Value>,
    data_off: u64,
    pub metadata: Map<String, Value>,
}

impl SafeTensors {
    pub fn open(path: &str) -> SafeTensors {
        let mut f = File::open(path).unwrap_or_else(|e| die(&format!("{}: {}", path, e)));
        let mut lenb = [0u8; 8];
        f.read_exact(&mut lenb)
            .unwrap_or_else(|e| die(&format!("{}: {}", path, e)));
        let n = u64::from_le_bytes(lenb);
        if n == 0 || n > 100_000_000 {
            die(&format!("{}: implausible header length {}", path, n));
        }
        let mut hb = vec![0u8; n as usize];
        f.read_exact(&mut hb)
            .unwrap_or_else(|e| die(&format!("{}: {}", path, e)));
        let v: Value = serde_json::from_slice(&hb)
            .unwrap_or_else(|e| die(&format!("{}: bad header json: {}", path, e)));
        let mut header = v
            .as_object()
            .unwrap_or_else(|| die(&format!("{}: header is not an object", path)))
            .clone();
        let metadata = header
            .remove("__metadata__")
            .and_then(|m| m.as_object().cloned())
            .unwrap_or_default();
        SafeTensors {
            path: path.to_string(),
            header,
            data_off: 8 + n,
            metadata,
        }
    }

    pub fn names(&self) -> Vec<String> {
        self.header.keys().cloned().collect()
    }

    pub fn has(&self, name: &str) -> bool {
        self.header.contains_key(name)
    }

    fn entry(&self, name: &str) -> (&str, Vec<usize>, u64, u64) {
        let e = self
            .header
            .get(name)
            .unwrap_or_else(|| die(&format!("{}: no tensor named {}", self.path, name)));
        let o = e
            .as_object()
            .unwrap_or_else(|| die(&format!("{}: {} is not an object", self.path, name)));
        let dt = o
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| die(&format!("{}: {} has no dtype", self.path, name)));
        let dims: Vec<usize> = o
            .get("shape")
            .and_then(|v| v.as_array())
            .unwrap_or_else(|| die(&format!("{}: {} has no shape", self.path, name)))
            .iter()
            .map(|x| x.as_u64().unwrap_or(0) as usize)
            .collect();
        let offs = o
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .unwrap_or_else(|| die(&format!("{}: {} has no data_offsets", self.path, name)));
        if offs.len() != 2 {
            die(&format!("{}: {} data_offsets must have 2 entries", self.path, name));
        }
        let a = offs[0].as_u64().unwrap_or(0);
        let b = offs[1].as_u64().unwrap_or(0);
        (dt, dims, a, b - a)
    }

    pub fn dtype(&self, name: &str) -> String {
        self.entry(name).0.to_string()
    }

    pub fn dims(&self, name: &str) -> Vec<usize> {
        self.entry(name).1
    }

    pub fn raw(&self, name: &str) -> Vec<u8> {
        let (_, _, off, len) = self.entry(name);
        let mut f = File::open(&self.path).unwrap_or_else(|e| die(&format!("{}: {}", self.path, e)));
        f.seek(SeekFrom::Start(self.data_off + off))
            .unwrap_or_else(|e| die(&format!("{}: {}", self.path, e)));
        let mut b = vec![0u8; len as usize];
        f.read_exact(&mut b)
            .unwrap_or_else(|e| die(&format!("{}: {}", self.path, e)));
        b
    }

    pub fn f32(&self, name: &str) -> Vec<f32> {
        let (dt, dims, _, _) = self.entry(name);
        let b = self.raw(name);
        let want: usize = dims.iter().product();
        let v: Vec<f32> = match dt {
            "F32" => b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            "F16" => b
                .chunks_exact(2)
                .map(|c| h2f(u16::from_le_bytes(c.try_into().unwrap())))
                .collect(),
            "BF16" => b
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16))
                .collect(),
            other => die(&format!("{}: {} has dtype {}", self.path, name, other)),
        };
        assert_eq!(
            v.len(),
            want,
            "{}: {} shape {:?} covers {} elements but {} were read",
            self.path,
            name,
            dims,
            want,
            v.len()
        );
        v
    }
}
