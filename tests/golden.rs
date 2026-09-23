use laya_rust::engine::{BackendId, Engine, Req};
use laya_rust::{backend, model_spec, systemone};
use serde_json::Value;
use std::path::PathBuf;

const ATOL: f32 = 1e-3;
const RTOL: f32 = 1e-4;
const PTOL: f64 = 1e-3;

fn within(got: f32, want: f32) -> bool {
    ((got - want).abs() as f64) <= ATOL as f64 + RTOL as f64 * (want.abs() as f64)
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn model_dir() -> Option<PathBuf> {
    let d = std::env::var("LAYA_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root().join("models/laya"));
    let gguf = laya_rust::gguf_path(d.to_str()?);
    PathBuf::from(gguf).exists().then_some(d)
}

fn f32s(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap_or_else(|| panic!("expected array, got {}", v))
        .iter()
        .map(|x| x.as_f64().expect("float") as f32)
        .collect()
}

fn i64s(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap_or_else(|| panic!("expected array, got {}", v))
        .iter()
        .map(|x| x.as_i64().expect("int"))
        .collect()
}

fn fixtures() -> Vec<PathBuf> {
    let dir = repo_root().join("tests/fixtures");
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {}", dir.display(), e))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    v.sort();
    v
}

fn answer_numbers(a: &Value) -> Vec<(String, f64)> {
    let mut out: Vec<(String, f64)> = Vec::new();
    let mut push = |k: &str, v: Option<f64>| {
        if let Some(v) = v {
            out.push((k.to_string(), v));
        }
    };
    push("confidence", a.get("confidence").and_then(|x| x.as_f64()));
    push(
        "act_probability",
        a["rl_agent"].get("act_probability").and_then(|x| x.as_f64()),
    );
    push("noul", a.get("noul").and_then(|x| x.as_f64()));
    push("score", a.get("score").and_then(|x| x.as_f64()));
    if let Some(p) = a.get("probabilities").and_then(|x| x.as_object()) {
        for (k, v) in p {
            push(
                &format!("probabilities.{}", k),
                v.as_f64(),
            );
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[test]
fn golden_forward_and_answers() {
    let Some(dir) = model_dir() else {
        eprintln!(
            "SKIP golden: no model. Set LAYA_MODEL_DIR to a directory or a .gguf file, or place the GGUF in models/laya"
        );
        return;
    };
    let backend_id = std::env::var("LAYA_TEST_BACKEND")
        .ok()
        .and_then(|s| BackendId::parse(&s))
        .unwrap_or(BackendId::Auto);
    let spec = model_spec(dir.to_str().unwrap(), backend_id, false);
    let mut engine = backend::open(&spec);
    let info = engine.info();
    eprintln!(
        "golden: replaying fixtures with backend={} device={:?} dtype={}",
        info.id.name(),
        info.device,
        info.dtype.name()
    );

    let files = fixtures();
    assert!(!files.is_empty(), "no fixtures in tests/fixtures");

    let (mut worst_logit, mut worst_logit_rel) = (0f32, 0f32);
    let (mut worst_act, mut worst_act_rel) = (0f32, 0f32);
    let mut worst_prob = 0f64;
    let mut checked_forwards = 0usize;
    let mut checked_rows = 0usize;

    for f in &files {
        let doc: Value =
            serde_json::from_str(&std::fs::read_to_string(f).unwrap()).expect("fixture json");
        let name = doc["name"].as_str().unwrap().to_string();
        let recorded_by = doc["backend"].as_str().unwrap_or("?").to_string();
        let req: Value = doc["request"].clone();
        let (_, questions) = systemone::parse_request(&req);

        for fw in doc["forwards"].as_array().expect("forwards") {
            let (n, l, k) = (
                fw["n"].as_u64().unwrap() as usize,
                fw["l"].as_u64().unwrap() as usize,
                fw["k"].as_u64().unwrap() as usize,
            );
            let ids = i64s(&fw["ids"]);
            let att = i64s(&fw["att"]);
            let mpos = i64s(&fw["mpos"]);
            let mmask = i64s(&fw["mmask"]);
            let qtype = i64s(&fw["qtype"]);
            let want_logits = f32s(&fw["logits"]);
            let want_act = f32s(&fw["act"]);

            let out = engine.forward(&Req {
                n,
                l,
                k,
                ids: &ids,
                att: &att,
                mpos: &mpos,
                mmask: &mmask,
                qtype: &qtype,
            });
            checked_forwards += 1;
            assert_eq!(out.logits.len(), want_logits.len(), "{}: logits len", name);
            assert_eq!(out.act.len(), want_act.len(), "{}: act len", name);

            for (i, (got, want)) in out.logits.iter().zip(&want_logits).enumerate() {
                let d = (got - want).abs();
                if d > worst_logit {
                    worst_logit = d;
                    worst_logit_rel = d / want.abs().max(1e-30);
                }
                assert!(
                    within(*got, *want),
                    "{}: logits[{}] got {} want {} (delta {:.3e}, recorded by {})",
                    name,
                    i,
                    got,
                    want,
                    d,
                    recorded_by
                );
            }
            for (i, (got, want)) in out.act.iter().zip(&want_act).enumerate() {
                let d = (got - want).abs();
                if d > worst_act {
                    worst_act = d;
                    worst_act_rel = d / want.abs().max(1e-30);
                }
                assert!(
                    within(*got, *want),
                    "{}: act[{}] got {} want {} (delta {:.3e})",
                    name,
                    i,
                    got,
                    want,
                    d
                );
            }

            for (row, r) in fw["rows"].as_array().expect("rows").iter().enumerate() {
                let id = r["id"].as_str().unwrap();
                let markers = r["markers"].as_u64().unwrap() as usize;
                let qt = r["qtype"].as_i64().unwrap();
                let q = questions
                    .iter()
                    .find(|q| q.id == id)
                    .unwrap_or_else(|| panic!("{}: no question {}", name, id));
                let raw = &out.logits[row * k..row * k + markers];
                let t = systemone::temperature(qt, markers, &info.calibration);
                let probs = systemone::softmax_t(raw, t);
                let a = &out.act[row * 2..row * 2 + 2];
                let m = a.iter().cloned().fold(f32::MIN, f32::max);
                let e0 = (a[0] - m).exp();
                let e1 = (a[1] - m).exp();
                let got = systemone::answer(q, &probs, e0 / (e0 + e1));
                let want = &doc["answers"][id];
                checked_rows += 1;

                assert_eq!(got["type"], want["type"], "{}:{} type", name, id);
                let gn = answer_numbers(&got);
                let wn = answer_numbers(want);
                let gk: Vec<&String> = gn.iter().map(|(k, _)| k).collect();
                let wk: Vec<&String> = wn.iter().map(|(k, _)| k).collect();
                assert_eq!(gk, wk, "{}:{} answer fields", name, id);
                for ((field, g), (_, w)) in gn.iter().zip(wn.iter()) {
                    let d = (g - w).abs();
                    worst_prob = worst_prob.max(d);
                    assert!(
                        d <= PTOL,
                        "{}:{} {} got {} want {} (delta {:.3e})",
                        name,
                        id,
                        field,
                        g,
                        w,
                        d
                    );
                }
                if got["type"] == "choice" {
                    assert_eq!(
                        got["choice"], want["choice"],
                        "{}:{} selected a different option",
                        name, id
                    );
                }
            }
        }
    }

    eprintln!(
        "golden: {} forward(s), {} scored row(s) via {}; raw worst |dlogit|={:.3e} (rel {:.2e}), |dact|={:.3e} (rel {:.2e}); calibrated worst |d|={:.3e} (limit {:.1e})",
        checked_forwards,
        checked_rows,
        info.id.name(),
        worst_logit,
        worst_logit_rel,
        worst_act,
        worst_act_rel,
        worst_prob,
        PTOL
    );
    assert!(checked_forwards > 0, "no forwards were checked");
}

#[test]
fn fixture_floats_round_trip_bit_exactly() {
    let samples = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        0.1,
        -12345.678,
        1e-30,
        -1e30,
        f32::MIN,
        f32::MAX,
        f32::MIN_POSITIVE,
        0.3535533845424652,
        3.1415927,
    ];
    let json = laya_rust::floats(&samples);
    let back: Vec<f32> = json
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(samples.len(), back.len());
    for (i, (a, b)) in samples.iter().zip(&back).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "sample {} round-tripped {} -> {}",
            i,
            a,
            b
        );
    }
}
