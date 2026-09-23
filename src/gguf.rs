use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

pub const MAGIC: u32 = 0x4655_4747;
pub const VERSION: u32 = 3;
pub const ALIGNMENT: u64 = 32;

pub const T_F32: u32 = 0;
pub const T_F16: u32 = 1;
pub const T_BF16: u32 = 30;

pub const FT_ALL_F32: u32 = 0;
pub const FT_MOSTLY_F16: u32 = 1;

#[derive(Clone, Debug)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn type_id(&self) -> u32 {
        match self {
            Value::U8(_) => 0,
            Value::I8(_) => 1,
            Value::U16(_) => 2,
            Value::I16(_) => 3,
            Value::U32(_) => 4,
            Value::I32(_) => 5,
            Value::F32(_) => 6,
            Value::Bool(_) => 7,
            Value::Str(_) => 8,
            Value::Array(_) => 9,
            Value::U64(_) => 10,
            Value::I64(_) => 11,
            Value::F64(_) => 12,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U8(v) => Some(*v as u64),
            Value::U16(v) => Some(*v as u64),
            Value::U32(v) => Some(*v as u64),
            Value::U64(v) => Some(*v),
            Value::I8(v) => Some(*v as u64),
            Value::I16(v) => Some(*v as u64),
            Value::I32(v) => Some(*v as u64),
            Value::I64(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Value::F32(v) => Some(*v),
            Value::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_f32s(&self) -> Vec<f32> {
        match self {
            Value::Array(a) => a.iter().filter_map(|v| v.as_f32()).collect(),
            _ => Vec::new(),
        }
    }

    pub fn as_strs(&self) -> Vec<String> {
        match self {
            Value::Array(a) => a
                .iter()
                .map(|v| v.as_str().unwrap_or_default().to_string())
                .collect(),
            _ => Vec::new(),
        }
    }
}

pub fn elem_size(ty: u32) -> u64 {
    match ty {
        T_F32 => 4,
        T_F16 | T_BF16 => 2,
        other => die(&format!("gguf: unsupported tensor type {}", other)),
    }
}

pub fn type_name(ty: u32) -> &'static str {
    match ty {
        T_F32 => "F32",
        T_F16 => "F16",
        T_BF16 => "BF16",
        _ => "?",
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: u32,
    pub offset: u64,
}

impl TensorInfo {
    pub fn elems(&self) -> u64 {
        self.dims.iter().product()
    }

    pub fn nbytes(&self) -> u64 {
        self.elems() * elem_size(self.ty)
    }
}

pub struct Header {
    pub kv: Vec<(String, Value)>,
    pub tensors: Vec<TensorInfo>,
    pub data_off: u64,
}

impl Header {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn str_or(&self, key: &str, d: &str) -> String {
        self.get(key)
            .and_then(|v| v.as_str())
            .unwrap_or(d)
            .to_string()
    }

    pub fn u64_or(&self, key: &str, d: u64) -> u64 {
        self.get(key).and_then(|v| v.as_u64()).unwrap_or(d)
    }

    pub fn f32_or(&self, key: &str, d: f32) -> f32 {
        self.get(key).and_then(|v| v.as_f32()).unwrap_or(d)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

fn die(m: &str) -> ! {
    eprintln!("{}", m);
    std::process::exit(2);
}

fn align_up(x: u64) -> u64 {
    x + (ALIGNMENT - (x % ALIGNMENT)) % ALIGNMENT
}

struct Rd {
    f: File,
    pos: u64,
    len: u64,
}

impl Rd {
    fn open(path: &str) -> Rd {
        let f = File::open(path).unwrap_or_else(|e| die(&format!("{}: {}", path, e)));
        let len = f
            .metadata()
            .unwrap_or_else(|e| die(&format!("{}: {}", path, e)))
            .len();
        Rd { f, pos: 0, len }
    }

    fn take(&mut self, n: u64, ctx: &str) -> Vec<u8> {
        if self.pos + n > self.len {
            die(&format!(
                "gguf: {}: read of {} bytes at {} passes end of file ({})",
                ctx, n, self.pos, self.len
            ));
        }
        let mut v = vec![0u8; n as usize];
        self.f
            .read_exact(&mut v)
            .unwrap_or_else(|e| die(&format!("gguf: {}: {}", ctx, e)));
        self.pos += n;
        v
    }

    fn u8(&mut self, ctx: &str) -> u8 {
        self.take(1, ctx)[0]
    }

    fn u16(&mut self, ctx: &str) -> u16 {
        u16::from_le_bytes(self.take(2, ctx).try_into().unwrap())
    }

    fn u32(&mut self, ctx: &str) -> u32 {
        u32::from_le_bytes(self.take(4, ctx).try_into().unwrap())
    }

    fn u64(&mut self, ctx: &str) -> u64 {
        u64::from_le_bytes(self.take(8, ctx).try_into().unwrap())
    }

    fn i8(&mut self, ctx: &str) -> i8 {
        self.u8(ctx) as i8
    }

    fn i16(&mut self, ctx: &str) -> i16 {
        self.u16(ctx) as i16
    }

    fn i32(&mut self, ctx: &str) -> i32 {
        self.u32(ctx) as i32
    }

    fn i64(&mut self, ctx: &str) -> i64 {
        self.u64(ctx) as i64
    }

    fn f32(&mut self, ctx: &str) -> f32 {
        f32::from_bits(self.u32(ctx))
    }

    fn f64(&mut self, ctx: &str) -> f64 {
        f64::from_bits(self.u64(ctx))
    }

    fn string(&mut self, ctx: &str) -> String {
        let n = self.u64(ctx);
        let b = self.take(n, ctx);
        String::from_utf8(b).unwrap_or_else(|_| die(&format!("gguf: {}: invalid utf-8", ctx)))
    }
}

fn read_value(r: &mut Rd, ty: u32, ctx: &str) -> Value {
    match ty {
        0 => Value::U8(r.u8(ctx)),
        1 => Value::I8(r.i8(ctx)),
        2 => Value::U16(r.u16(ctx)),
        3 => Value::I16(r.i16(ctx)),
        4 => Value::U32(r.u32(ctx)),
        5 => Value::I32(r.i32(ctx)),
        6 => Value::F32(r.f32(ctx)),
        7 => Value::Bool(r.u8(ctx) != 0),
        8 => Value::Str(r.string(ctx)),
        9 => {
            let et = r.u32(ctx);
            let n = r.u64(ctx);
            let guard = n.saturating_mul(8) + 1;
            if guard > r.len {
                die(&format!(
                    "gguf: {}: array of {} elements is not plausible in {} bytes",
                    ctx, n, r.len
                ));
            }
            let mut v = Vec::with_capacity(n as usize);
            for _ in 0..n {
                v.push(read_value(r, et, ctx));
            }
            Value::Array(v)
        }
        10 => Value::U64(r.u64(ctx)),
        11 => Value::I64(r.i64(ctx)),
        12 => Value::F64(r.f64(ctx)),
        other => die(&format!(
            "gguf: {}: unknown metadata value type {}",
            ctx, other
        )),
    }
}

pub fn read(path: &str) -> Header {
    let mut r = Rd::open(path);
    let magic = r.u32(path);
    if magic != MAGIC {
        die(&format!(
            "{}: not a GGUF file (magic {:#010x}, want {:#010x})",
            path, magic, MAGIC
        ));
    }
    let version = r.u32(path);
    if version != VERSION {
        die(&format!("{}: GGUF version {}, want {}", path, version, VERSION));
    }
    let tensor_count = r.u64(path);
    let kv_count = r.u64(path);
    if tensor_count.saturating_mul(64) + kv_count.saturating_mul(16) > r.len {
        die(&format!(
            "{}: header claims {} tensors and {} metadata entries in {} bytes",
            path, tensor_count, kv_count, r.len
        ));
    }
    let mut kv = Vec::with_capacity(kv_count as usize);
    for _ in 0..kv_count {
        let key = r.string(path);
        let ty = r.u32(path);
        let val = read_value(&mut r, ty, &key);
        kv.push((key, val));
    }
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = r.string(path);
        let nd = r.u32(path);
        if nd == 0 || nd > 4 {
            die(&format!("gguf: tensor {} has {} dimensions", name, nd));
        }
        let mut dims = Vec::with_capacity(nd as usize);
        for _ in 0..nd {
            dims.push(r.u64(&name));
        }
        let ty = r.u32(&name);
        let offset = r.u64(&name);
        tensors.push(TensorInfo {
            name,
            dims,
            ty,
            offset,
        });
    }
    let data_off = align_up(r.pos);
    if data_off > r.len {
        die(&format!("{}: tensor data starts past end of file", path));
    }
    for t in &tensors {
        if t.offset % ALIGNMENT != 0 {
            die(&format!(
                "gguf: tensor {} offset {} is not a multiple of {}",
                t.name, t.offset, ALIGNMENT
            ));
        }
        if data_off + t.offset + t.nbytes() > r.len {
            die(&format!("gguf: tensor {} data is outside the file", t.name));
        }
    }
    Header {
        kv,
        tensors,
        data_off,
    }
}

pub fn key_ok(k: &str) -> bool {
    if k.is_empty() || k.len() > 65535 || !k.is_ascii() {
        return false;
    }
    k.split('.').all(|seg| {
        !seg.is_empty()
            && seg
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
    })
}

pub struct Writer {
    kv: Vec<(String, Value)>,
    tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
}

impl Default for Writer {
    fn default() -> Self {
        Self::new()
    }
}

impl Writer {
    pub fn new() -> Writer {
        Writer {
            kv: Vec::new(),
            tensors: Vec::new(),
        }
    }

    pub fn kv(&mut self, key: &str, v: Value) -> &mut Self {
        if !key_ok(key) {
            die(&format!("gguf: illegal metadata key {:?}", key));
        }
        self.kv.push((key.to_string(), v));
        self
    }

    pub fn tensor(&mut self, name: &str, dims: &[u64], ty: u32, data: Vec<u8>) -> &mut Self {
        if name.is_empty() || name.len() > 64 || !name.is_ascii() {
            die(&format!("gguf: illegal tensor name {:?}", name));
        }
        if dims.is_empty() || dims.len() > 4 {
            die(&format!(
                "gguf: tensor {} has {} dimensions",
                name,
                dims.len()
            ));
        }
        let elems: u64 = dims.iter().product();
        let want = elems * elem_size(ty);
        if want != data.len() as u64 {
            die(&format!(
                "gguf: tensor {} is {:?} {} = {} bytes but {} were given",
                name,
                dims,
                type_name(ty),
                want,
                data.len()
            ));
        }
        if self.tensors.iter().any(|t| t.0 == name) {
            die(&format!("gguf: duplicate tensor name {}", name));
        }
        self.tensors.push((name.to_string(), dims.to_vec(), ty, data));
        self
    }

    pub fn write(&self, path: &str) {
        let mut head = Vec::new();
        put_u32(&mut head, MAGIC);
        put_u32(&mut head, VERSION);
        put_u64(&mut head, self.tensors.len() as u64);
        put_u64(&mut head, self.kv.len() as u64);
        let mut seen = HashSet::new();
        for (k, v) in &self.kv {
            if !seen.insert(k.clone()) {
                die(&format!("gguf: duplicate metadata key {:?}", k));
            }
            put_string(&mut head, k);
            put_u32(&mut head, v.type_id());
            put_value(&mut head, v);
        }
        let mut off = 0u64;
        for (name, dims, ty, data) in &self.tensors {
            put_string(&mut head, name);
            put_u32(&mut head, dims.len() as u32);
            for d in dims {
                put_u64(&mut head, *d);
            }
            put_u32(&mut head, *ty);
            put_u64(&mut head, off);
            off = align_up(off + data.len() as u64);
        }
        let data_off = align_up(head.len() as u64);

        let mut out = head;
        out.resize(data_off as usize, 0);
        for (_, _, _, data) in &self.tensors {
            out.extend_from_slice(data);
            out.resize(align_up(out.len() as u64) as usize, 0);
        }
        let total = data_off + off;
        if out.len() as u64 != total {
            die(&format!(
                "gguf: wrote {} bytes but planned {}",
                out.len(),
                total
            ));
        }
        let mut f = File::create(path).unwrap_or_else(|e| die(&format!("{}: {}", path, e)));
        f.write_all(&out)
            .unwrap_or_else(|e| die(&format!("{}: {}", path, e)));
        f.sync_all().ok();
    }
}

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_string(b: &mut Vec<u8>, s: &str) {
    put_u64(b, s.len() as u64);
    b.extend_from_slice(s.as_bytes());
}

fn put_value(b: &mut Vec<u8>, v: &Value) {
    match v {
        Value::U8(x) => b.push(*x),
        Value::I8(x) => b.push(*x as u8),
        Value::U16(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::I16(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::U32(x) => put_u32(b, *x),
        Value::I32(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::F32(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::Bool(x) => b.push(*x as u8),
        Value::Str(s) => put_string(b, s),
        Value::Array(a) => {
            let et = a.first().map(|x| x.type_id()).unwrap_or(0);
            for x in a {
                if x.type_id() != et {
                    die("gguf: array elements must share one type");
                }
            }
            put_u32(b, et);
            put_u64(b, a.len() as u64);
            for x in a {
                put_value(b, x);
            }
        }
        Value::U64(x) => put_u64(b, *x),
        Value::I64(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::F64(x) => b.extend_from_slice(&x.to_le_bytes()),
    }
}

pub fn as_f16_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 2);
    for x in v {
        b.extend_from_slice(&crate::weights::f2h(*x).to_le_bytes());
    }
    b
}

pub fn as_f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

pub fn read_tensor_bytes(path: &str, h: &Header, name: &str) -> Vec<u8> {
    let t = h
        .tensor(name)
        .unwrap_or_else(|| die(&format!("gguf: no tensor named {}", name)));
    let mut f = File::open(path).unwrap_or_else(|e| die(&format!("{}: {}", path, e)));
    f.seek(SeekFrom::Start(h.data_off + t.offset))
        .unwrap_or_else(|e| die(&format!("{}: {}", path, e)));
    let n = t.nbytes() as usize;
    let mut b = vec![0u8; n];
    f.read_exact(&mut b)
        .unwrap_or_else(|e| die(&format!("{}: {}", path, e)));
    b
}

pub fn decode_f32(bytes: &[u8], ty: u32, what: &str) -> Vec<f32> {
    match ty {
        T_F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        T_F16 => bytes
            .chunks_exact(2)
            .map(|c| crate::weights::h2f(u16::from_le_bytes(c.try_into().unwrap())))
            .collect(),
        T_BF16 => bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16))
            .collect(),
        other => die(&format!(
            "{}: tensor type {} ({}) cannot be decoded",
            what,
            other,
            type_name(other)
        )),
    }
}
