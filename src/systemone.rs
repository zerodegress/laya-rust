use crate::tokenizer::Tokenizer;
use serde_json::{Value, json};

pub const MAX_LEN: usize = 512;
pub const HEAD_MAX_LEN: usize = 192;

#[derive(Clone, Debug)]
pub struct Calibration {
    pub temperature: [f32; 3],
    pub by_options: Vec<(String, f32)>,
}

impl Calibration {
    pub fn uniform() -> Calibration {
        Calibration {
            temperature: [1.0; 3],
            by_options: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct Question {
    pub id: String,
    pub ty: String,
    pub qtype: i64,
    pub instructions: String,
    pub crit: Value,
}

pub fn qtype_of(t: &str) -> i64 {
    match t {
        "choice" => 0,
        "score" => 1,
        _ => 2,
    }
}

pub fn qname(q: i64) -> &'static str {
    match q {
        0 => "choice",
        1 => "score",
        _ => "noul",
    }
}

fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        Value::Bool(b) => (if *b { "True" } else { "False" }).into(),
        Value::Number(n) => n.to_string(),
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(py_str).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(m) => format!(
            "{{{}}}",
            m.iter()
                .map(|(k, v)| format!("'{}': {}", k, py_str(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        v => serde_json::to_string(v).unwrap(),
    }
}

pub fn parse_request(req: &Value) -> Result<(Value, Vec<Question>), String> {
    let state = req.get("state").cloned().unwrap_or(Value::Null);
    let qs = req["questions"]
        .as_object()
        .ok_or_else(|| "missing \"questions\"".to_string())?;
    let mut out = Vec::new();
    for (id, q) in qs {
        let ty = q["type"]
            .as_str()
            .ok_or_else(|| format!("question {}: missing \"type\"", id))?
            .to_string();
        if !matches!(ty.as_str(), "choice" | "score" | "noul") {
            return Err(format!("question {}: bad type {}", id, ty));
        }
        let instructions = match &q["instructions"] {
            Value::String(s) => s.clone(),
            Value::Null => return Err(format!("question {}: missing \"instructions\"", id)),
            v => serde_json::to_string(v).unwrap(),
        };
        let mut crit = q.get("criteria").cloned().unwrap_or(Value::Null);
        if ty == "choice" && crit.is_array() {
            let mut m = serde_json::Map::new();
            for c in crit.as_array().unwrap() {
                m.insert(py_str(c), Value::Null);
            }
            crit = Value::Object(m);
        }
        out.push(Question {
            id: id.clone(),
            qtype: qtype_of(&ty),
            ty,
            instructions,
            crit,
        });
    }
    Ok((state, out))
}

pub fn render_options(q: &Question) -> Vec<String> {
    match q.ty.as_str() {
        "choice" => match q.crit.as_object() {
            Some(m) => m
                .iter()
                .map(|(k, v)| {
                    if v.is_null() {
                        k.clone()
                    } else {
                        format!("{}: {}", k, py_str(v))
                    }
                })
                .collect(),
            None => Vec::new(),
        },
        "score" => q
            .crit
            .as_array()
            .map(|a| {
                a.iter()
                    .enumerate()
                    .map(|(i, c)| format!("level {}: {}", i, py_str(c)))
                    .collect()
            })
            .unwrap_or_default(),
        _ => {
            let m = q.crit.as_object();
            let get = |k: &str, d: &str| -> String {
                m.and_then(|m| m.get(k))
                    .filter(|v| !v.is_null())
                    .map(py_str)
                    .unwrap_or_else(|| d.to_string())
            };
            vec![
                format!(
                    "false: {}",
                    get("false", "no, the statement does not hold")
                ),
                format!("true: {}", get("true", "yes, the statement holds")),
            ]
        }
    }
}

pub fn build_sequence(tok: &Tokenizer, state: &Value, q: &Question) -> (Vec<i64>, Vec<i64>) {
    let opts = render_options(q);
    let ins = q.instructions.replace("[MASK]", " ");
    let head_ids = tok.encode(&format!("{} question: {}", q.ty, ins), false);
    let mut opt_ids: Vec<Vec<i64>> = Vec::new();
    for o in &opts {
        let mut ids = tok.encode(&format!(" {}", o.replace("[MASK]", " ")), false);
        ids.truncate(48);
        let mut v = vec![tok.mask];
        v.extend(ids);
        opt_ids.push(v);
    }
    let sum = |v: &Vec<Vec<i64>>| -> i64 { v.iter().map(|o| o.len() as i64).sum() };
    let mut budget = HEAD_MAX_LEN as i64 - sum(&opt_ids);
    if budget < 16 {
        let per = std::cmp::max(
            4,
            (HEAD_MAX_LEN as i64 - 16) / std::cmp::max(1, opt_ids.len() as i64),
        );
        for o in opt_ids.iter_mut() {
            o.truncate(per as usize);
        }
        budget = HEAD_MAX_LEN as i64 - sum(&opt_ids);
    }
    let keep = std::cmp::max(8, budget) as usize;
    let head_ids: Vec<i64> = head_ids.into_iter().take(keep).collect();
    let mut ids = vec![tok.cls];
    ids.extend(head_ids);
    ids.push(tok.sep);
    let mut markers: Vec<i64> = Vec::new();
    for o in &opt_ids {
        markers.push(ids.len() as i64);
        ids.extend(o);
    }
    ids.push(tok.sep);
    let room = MAX_LEN.saturating_sub(ids.len() + 1);
    let st = tok.encode(&serialize_state(state).replace("[MASK]", " "), false);
    ids.extend(st.into_iter().take(room));
    ids.push(tok.sep);
    ids.truncate(MAX_LEN);
    let markers = markers.into_iter().filter(|m| *m < MAX_LEN as i64).collect();
    (ids, markers)
}

pub fn temperature(qtype: i64, k: usize, cal: &Calibration) -> f32 {
    let size = if k <= 2 {
        "2"
    } else if k <= 5 {
        "3-5"
    } else if k <= 10 {
        "6-10"
    } else {
        "11+"
    };
    let key = format!("{}:{}", qname(qtype), size);
    cal.by_options
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, v)| *v)
        .unwrap_or(cal.temperature[qtype as usize])
}

pub fn softmax_t(logits: &[f32], t: f32) -> Vec<f32> {
    let z: Vec<f32> = logits.iter().map(|x| x / t).collect();
    let m = z.iter().cloned().fold(f32::MIN, f32::max);
    let e: Vec<f32> = z.iter().map(|x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

pub fn confidence(p: &[f32]) -> f32 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let ent: f32 = p.iter().map(|x| -x * x.max(1e-12).ln()).sum();
    (1.0 - ent / (k as f32).ln()).max(0.0)
}

fn r4(x: f32) -> f32 {
    (x * 10000.0).round() / 10000.0
}

pub fn answer(q: &Question, p: &[f32], act_prob: f32) -> Value {
    let conf = r4(confidence(p));
    let ext = json!({ "act_probability": r4(act_prob) });
    match q.ty.as_str() {
        "choice" => {
            let keys: Vec<String> = q
                .crit
                .as_object()
                .map(|m| m.keys().cloned().collect())
                .unwrap_or_default();
            let best = p
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let probs: serde_json::Map<String, Value> = keys
                .iter()
                .zip(p.iter())
                .map(|(k, v)| (k.clone(), json!(r4(*v))))
                .collect();
            json!({
                "type": "choice",
                "choice": keys.get(best).cloned().unwrap_or_default(),
                "probabilities": probs,
                "confidence": conf,
                "rl_agent": ext
            })
        }
        "score" => {
            let desc: Vec<Value> = q.crit.as_array().cloned().unwrap_or_default();
            let score: f32 = p.iter().enumerate().map(|(i, v)| i as f32 * v).sum();
            let legend: serde_json::Map<String, Value> = desc
                .iter()
                .enumerate()
                .map(|(i, c)| (i.to_string(), json!(py_str(c))))
                .collect();
            let probs: serde_json::Map<String, Value> = p
                .iter()
                .enumerate()
                .map(|(i, v)| (i.to_string(), json!(r4(*v))))
                .collect();
            json!({
                "type": "score",
                "score": r4(score),
                "legend": legend,
                "probabilities": probs,
                "confidence": conf,
                "rl_agent": ext
            })
        }
        _ => json!({
            "type": "noul",
            "noul": r4(*p.get(1).unwrap_or(&0.0)),
            "rl_agent": ext
        }),
    }
}
