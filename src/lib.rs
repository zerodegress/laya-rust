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
pub const MODEL_ENV: &str = "LAYA_MODEL";

pub fn model_path(arg: Option<&str>) -> String {
    arg.map(|s| s.to_string())
        .or_else(|| std::env::var(MODEL_ENV).ok().filter(|s| !s.is_empty()))
        .unwrap_or_else(|| DEFAULT_MODEL.to_string())
}

pub fn gguf_path(model: &str) -> String {
    if std::path::Path::new(model).is_file() {
        model.to_string()
    } else {
        format!("{}/{}", model.trim_end_matches('/'), GGUF_FILE)
    }
}

pub fn model_spec(model: &str, backend: engine::BackendId, quantized: bool) -> engine::ModelSpec {
    engine::ModelSpec {
        backend,
        base: gguf_path(model),
        quantized,
        quant_bits: 4,
        quant_group: 64,
    }
}

pub fn tokenizer_json(model: &str) -> String {
    let path = gguf_path(model);
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
