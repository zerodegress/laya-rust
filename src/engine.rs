pub struct Req<'a> {
    pub n: usize,
    pub l: usize,
    pub k: usize,
    pub ids: &'a [i64],
    pub att: &'a [i64],
    pub mpos: &'a [i64],
    pub mmask: &'a [i64],
    pub qtype: &'a [i64],
}

pub struct Out {
    pub n: usize,
    pub k: usize,
    pub logits: Vec<f32>,
    pub act: Vec<f32>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DType {
    F32,
    F16,
}

impl DType {
    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "fp32",
            DType::F16 => "fp16",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BackendId {
    Auto,
    Cuda,
    Mlx,
    Cpu,
}

impl BackendId {
    pub fn parse(s: &str) -> Option<BackendId> {
        match s {
            "auto" => Some(BackendId::Auto),
            "cuda" => Some(BackendId::Cuda),
            "mlx" => Some(BackendId::Mlx),
            "cpu" => Some(BackendId::Cpu),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            BackendId::Auto => "auto",
            BackendId::Cuda => "cuda",
            BackendId::Mlx => "mlx",
            BackendId::Cpu => "cpu",
        }
    }
}

pub struct ModelSpec {
    pub backend: BackendId,
    pub base: String,
    pub quantized: bool,
    pub quant_bits: i32,
    pub quant_group: i32,
}

pub struct EngineInfo {
    pub id: BackendId,
    pub device: String,
    pub dtype: DType,
    pub layers: usize,
    pub calibration: crate::systemone::Calibration,
}

pub trait Engine {
    fn info(&self) -> EngineInfo;
    fn forward(&mut self, req: &Req) -> Out;
}

impl<T: Engine + ?Sized> Engine for Box<T> {
    fn info(&self) -> EngineInfo {
        (**self).info()
    }

    fn forward(&mut self, req: &Req) -> Out {
        (**self).forward(req)
    }
}
