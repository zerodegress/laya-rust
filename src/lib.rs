#[cfg(not(any(feature = "cuda", feature = "mlx", feature = "cpu")))]
compile_error!("enable at least one backend feature: cuda / mlx / cpu");

pub mod arch;
pub mod backend;
pub mod convert;
pub mod engine;
pub mod gguf;
#[cfg(any(feature = "cpu", feature = "mlx"))]
pub mod graph;
#[cfg(any(feature = "cpu", feature = "mlx"))]
pub mod ops;
pub mod systemone;
pub mod tokenizer;
pub mod weights;

use serde_json::Value;

pub const DEFAULT_MODEL: &str = "models/laya";
pub const GGUF_FILE: &str = "laya-f16.gguf";

pub fn model_spec(
    dir: &str,
    backend: engine::BackendId,
    quantized: bool,
) -> engine::ModelSpec {
    engine::ModelSpec {
        backend,
        base: format!("{}/{}", dir, GGUF_FILE),
        quantized,
        quant_bits: 4,
        quant_group: 64,
    }
}

pub fn tokenizer_json(dir: &str) -> String {
    let path = format!("{}/{}", dir, GGUF_FILE);
    let h = gguf::read(&path);
    match h.get("tokenizer.huggingface.json").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => {
            eprintln!("{}: no embedded tokenizer.huggingface.json", path);
            std::process::exit(2);
        }
    }
}

pub fn floats(v: &[f32]) -> Value {
    Value::Array(
        v.iter()
            .map(|x| serde_json::Number::from_f64(*x as f64).map_or(Value::Null, Value::Number))
            .collect(),
    )
}
