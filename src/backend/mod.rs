use crate::engine::{BackendId, Engine, ModelSpec};

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "cpu")]
pub mod cpu;

#[cfg(feature = "mlx")]
pub mod mlx;

pub const ALL: [BackendId; 3] = [BackendId::Cuda, BackendId::Mlx, BackendId::Cpu];

fn die(m: &str) -> ! {
    eprintln!("{}", m);
    std::process::exit(2);
}

pub fn compiled(id: BackendId) -> bool {
    match id {
        BackendId::Cuda => cfg!(feature = "cuda"),
        BackendId::Mlx => cfg!(feature = "mlx"),
        BackendId::Cpu => cfg!(feature = "cpu"),
        BackendId::Auto => true,
    }
}

pub fn available() -> Vec<&'static str> {
    ALL.iter().filter(|id| compiled(**id)).map(|id| id.name()).collect()
}

pub fn auto_id() -> BackendId {
    match ALL.iter().find(|id| compiled(**id)) {
        Some(id) => *id,
        None => die("no inference backend compiled in (enable a cargo feature: cuda / mlx / cpu)"),
    }
}

pub fn open(spec: &ModelSpec) -> Box<dyn Engine> {
    let id = if spec.backend == BackendId::Auto {
        auto_id()
    } else {
        spec.backend
    };
    if !compiled(id) {
        die(&format!(
            "backend {} is not compiled in (compiled: {})",
            id.name(),
            available().join(", ")
        ));
    }
    match id {
        BackendId::Cuda => open_cuda(spec),
        BackendId::Mlx => open_mlx(spec),
        BackendId::Cpu => open_cpu(spec),
        BackendId::Auto => unreachable!("auto is resolved above"),
    }
}

#[cfg(feature = "cuda")]
fn open_cuda(spec: &ModelSpec) -> Box<dyn Engine> {
    Box::new(cuda::CudaEngine::load(spec))
}

#[cfg(not(feature = "cuda"))]
fn open_cuda(_spec: &ModelSpec) -> Box<dyn Engine> {
    unreachable!("cuda is not compiled in")
}

#[cfg(feature = "mlx")]
fn open_mlx(spec: &ModelSpec) -> Box<dyn Engine> {
    Box::new(mlx::MlxEngine::load(spec))
}

#[cfg(not(feature = "mlx"))]
fn open_mlx(_spec: &ModelSpec) -> Box<dyn Engine> {
    unreachable!("mlx is not compiled in")
}

#[cfg(feature = "cpu")]
fn open_cpu(spec: &ModelSpec) -> Box<dyn Engine> {
    Box::new(cpu::CpuEngine::load(spec))
}

#[cfg(not(feature = "cpu"))]
fn open_cpu(_spec: &ModelSpec) -> Box<dyn Engine> {
    unreachable!("cpu is not compiled in")
}
