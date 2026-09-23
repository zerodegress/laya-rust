use crate::arch::*;
use crate::gguf::{self, Value, Writer};
use crate::weights::SafeTensors;
use serde_json::Value as J;

fn die(m: &str) -> ! {
    eprintln!("{}", m);
    std::process::exit(2);
}

fn read_json(path: &str) -> J {
    let s = std::fs::read_to_string(path).unwrap_or_else(|e| die(&format!("{}: {}", path, e)));
    serde_json::from_str(&s).unwrap_or_else(|e| die(&format!("{}: {}", path, e)))
}

pub fn rope_table(theta: f32, half: usize) -> Vec<f32> {
    let d = (2 * half) as f32;
    (0..half)
        .map(|j| 1.0f32 / theta.powf((2 * j) as f32 / d))
        .collect()
}

pub struct Stats {
    pub tensors: usize,
    pub bytes: u64,
    pub f16: bool,
    pub copied: usize,
    pub widened: usize,
    pub tokenizer_tokens: usize,
    pub tokenizer_merges: usize,
}

pub fn to_gguf(model: &str, out: &str, f16: bool) -> Stats {
    let st_path = format!("{}/model.safetensors", model);
    let st = SafeTensors::open(&st_path);
    let cfg = read_json(&format!("{}/encoder/config.json", model));
    let agent = read_json(&format!("{}/rl_agent_config.json", model));
    let tok_text = std::fs::read_to_string(format!("{}/tokenizer/tokenizer.json", model))
        .unwrap_or_else(|e| die(&format!("{}/tokenizer/tokenizer.json: {}", model, e)));
    let tok: J = serde_json::from_str(&tok_text).unwrap_or_else(|e| die(&format!("tokenizer: {}", e)));

    let get_num = |v: &J, k: &str| -> usize {
        v.get(k)
            .and_then(|x| x.as_u64())
            .unwrap_or_else(|| die(&format!("config.json: missing {}", k))) as usize
    };
    let hidden = get_num(&cfg, "hidden_size");
    let layers = get_num(&cfg, "num_hidden_layers");
    let heads = get_num(&cfg, "num_attention_heads");
    let inter = get_num(&cfg, "intermediate_size");
    let vocab = get_num(&cfg, "vocab_size");
    let eps = cfg
        .get("norm_eps")
        .or_else(|| cfg.get("layer_norm_eps"))
        .and_then(|x| x.as_f64())
        .unwrap_or_else(|| die("config.json: missing norm_eps")) as f32;
    let local = get_num(&cfg, "local_attention");
    let global_every = get_num(&cfg, "global_attn_every_n_layers");
    let rope = cfg
        .get("rope_parameters")
        .unwrap_or_else(|| die("config.json: missing rope_parameters"));

    for (k, want) in [
        ("hidden_size", H),
        ("num_hidden_layers", NLAYER),
        ("num_attention_heads", NH),
        ("intermediate_size", INTER),
        ("vocab_size", NVOCAB),
        ("global_attn_every_n_layers", GLOBAL_EVERY),
    ] {
        let got = get_num(&cfg, k);
        if got != want {
            die(&format!(
                "{}: config.json {} is {} but this build pins {}",
                model, k, got, want
            ));
        }
    }
    if local != 2 * (WINDOW as usize) {
        die(&format!(
            "{}: config.json local_attention is {} but WINDOW {} implies {}",
            model,
            local,
            WINDOW,
            2 * WINDOW
        ));
    }
    if eps.to_bits() != EPS.to_bits() {
        die(&format!(
            "{}: config.json norm_eps is {} but this build pins {}",
            model, eps, EPS
        ));
    }
    let theta_of = |k: &str| -> f32 {
        rope.get(k)
            .and_then(|x| x.get("rope_theta"))
            .and_then(|x| x.as_f64())
            .unwrap_or_else(|| die(&format!("config.json: missing rope_parameters.{}.rope_theta", k)))
            as f32
    };
    let theta_full = theta_of("full_attention");
    let theta_sliding = theta_of("sliding_attention");

    let mut w = Writer::new();
    w.kv("general.architecture", Value::Str("laya".into()));
    w.kv("general.name", Value::Str("Laya".into()));
    w.kv("general.license", Value::Str("Apache-2.0".into()));
    w.kv(
        "general.repo_url",
        Value::Str("https://huggingface.co/convaiinnovations/laya".into()),
    );
    w.kv(
        "general.description",
        Value::Str(
            "Non-autoregressive decision engine: ModernBERT-large encoder plus a decision head that scores option markers.".into(),
        ),
    );
    w.kv("general.file_type", Value::U32(if f16 { gguf::FT_MOSTLY_F16 } else { gguf::FT_ALL_F32 }));
    w.kv("general.alignment", Value::U32(gguf::ALIGNMENT as u32));

    w.kv("laya.weight_layout", Value::Str("ggml-canonical".into()));
    w.kv("laya.hidden_size", Value::U32(hidden as u32));
    w.kv("laya.num_layers", Value::U32(layers as u32));
    w.kv("laya.num_heads", Value::U32(heads as u32));
    w.kv("laya.head_dim", Value::U32((hidden / heads) as u32));
    w.kv("laya.intermediate_size", Value::U32(inter as u32));
    w.kv("laya.vocab_size", Value::U32(vocab as u32));
    w.kv("laya.attention_scale", Value::F32(SCALE));
    w.kv("laya.layer_norm_epsilon", Value::F32(EPS));
    w.kv("laya.sliding_window", Value::U32(WINDOW as u32));
    w.kv("laya.global_layer_every", Value::U32(global_every as u32));
    w.kv("laya.head_layers", Value::U32(HEAD_LAYERS as u32));
    w.kv("laya.head_intermediate_size", Value::U32(HEAD_INTER as u32));
    w.kv("laya.act_hidden_size", Value::U32(ACT_HIDDEN as u32));
    w.kv("laya.act_input_features", Value::U32(ACT_INPUT as u32));
    w.kv("laya.question_types", Value::U32(QTYPES as u32));
    w.kv("laya.rope_theta.full", Value::F32(theta_full));
    w.kv("laya.rope_theta.sliding", Value::F32(theta_sliding));

    for (k, d) in [
        ("max_len", 512u64),
        ("head_max_len", 192),
        ("max_prefixes", 6),
    ] {
        let v = agent.get(k).and_then(|x| x.as_u64()).unwrap_or(d);
        let key = format!("laya.{}", k);
        w.kv(&key, Value::U32(v as u32));
    }

    let temps: Vec<f32> = agent
        .get("temperature")
        .and_then(|x| x.as_array())
        .unwrap_or_else(|| die("rl_agent_config.json: missing temperature"))
        .iter()
        .map(|x| x.as_f64().unwrap_or(0.0) as f32)
        .collect();
    if temps.len() != QTYPES {
        die(&format!(
            "rl_agent_config.json: temperature has {} entries, want {}",
            temps.len(),
            QTYPES
        ));
    }
    for (i, t) in temps.iter().enumerate() {
        w.kv(&format!("laya.temperature.{}", i), Value::F32(*t));
    }
    let by_options = agent
        .get("temperature_by_options")
        .cloned()
        .unwrap_or(J::Object(Default::default()));
    let by_str = serde_json::to_string(&by_options).unwrap();
    w.kv("laya.temperature_by_options", Value::Str(by_str));

    let tokens = build_tokens(&tok);
    let merges = build_merges(&tok);
    let types = build_token_types(&tok, tokens.len());
    if tokens.len() != vocab {
        die(&format!(
            "tokenizer has {} tokens but vocab_size is {}",
            tokens.len(),
            vocab
        ));
    }
    let scores: Vec<f32> = types
        .iter()
        .map(|t| if *t == 1 { 0.0 } else { -1000.0 })
        .collect();
    w.kv("tokenizer.ggml.model", Value::Str("gpt2".into()));
    w.kv(
        "tokenizer.ggml.tokens",
        Value::Array(tokens.iter().map(|s| Value::Str(s.clone())).collect()),
    );
    w.kv(
        "tokenizer.ggml.merges",
        Value::Array(merges.iter().map(|s| Value::Str(s.clone())).collect()),
    );
    w.kv(
        "tokenizer.ggml.token_type",
        Value::Array(types.iter().map(|t| Value::I32(*t)).collect()),
    );
    w.kv(
        "tokenizer.ggml.scores",
        Value::Array(scores.iter().map(|s| Value::F32(*s)).collect()),
    );
    w.kv("tokenizer.ggml.add_space_prefix", Value::Bool(false));
    let tokid = |name: &str| -> u32 {
        tok.get("added_tokens")
            .and_then(|x| x.as_array())
            .and_then(|a| {
                a.iter()
                    .find(|t| t.get("content").and_then(|c| c.as_str()) == Some(name))
            })
            .and_then(|t| t.get("id"))
            .and_then(|x| x.as_u64())
            .unwrap_or_else(|| die(&format!("tokenizer.json: no token {}", name))) as u32
    };
    w.kv("tokenizer.ggml.bos_token_id", Value::U32(tokid("[CLS]")));
    w.kv("tokenizer.ggml.eos_token_id", Value::U32(tokid("[SEP]")));
    w.kv("tokenizer.ggml.cls_token_id", Value::U32(tokid("[CLS]")));
    w.kv("tokenizer.ggml.mask_token_id", Value::U32(tokid("[MASK]")));
    w.kv("tokenizer.ggml.unknown_token_id", Value::U32(tokid("[UNK]")));
    w.kv("tokenizer.ggml.padding_token_id", Value::U32(tokid("[PAD]")));
    w.kv("tokenizer.huggingface.json", Value::Str(tok_text));

    let mut copied = 0usize;
    let mut widened = 0usize;
    for name in weight_names() {
        let (data, dims, ty) = if name == ROPE_FREQ_FULL || name == ROPE_FREQ_SLIDING {
            let theta = if name == ROPE_FREQ_FULL { theta_full } else { theta_sliding };
            let v = rope_table(theta, HD / 2);
            widened += 1;
            (
                gguf::as_f32_bytes(&v),
                vec![(HD / 2) as u64],
                gguf::T_F32,
            )
        } else {
            if !st.has(&name) {
                die(&format!("{}: missing tensor {}", st_path, name));
            }
            let dims = st.dims(&name);
            let dt = st.dtype(&name);
            let gdims: Vec<u64> = dims.iter().rev().map(|x| *x as u64).collect();
            if f16 && dt == "F16" {
                copied += 1;
                (st.raw(&name), gdims, gguf::T_F16)
            } else if !f16 && dt == "F32" {
                copied += 1;
                (st.raw(&name), gdims, gguf::T_F32)
            } else {
                widened += 1;
                let v = st.f32(&name);
                (
                    if f16 { gguf::as_f16_bytes(&v) } else { gguf::as_f32_bytes(&v) },
                    gdims,
                    if f16 { gguf::T_F16 } else { gguf::T_F32 },
                )
            }
        };
        w.tensor(&name, &dims, ty, data);
    }

    w.write(out);
    Stats {
        tensors: weight_names().len(),
        bytes: std::fs::metadata(out).map(|m| m.len()).unwrap_or(0),
        f16,
        copied,
        widened,
        tokenizer_tokens: tokens.len(),
        tokenizer_merges: merges.len(),
    }
}

fn build_tokens(tok: &J) -> Vec<String> {
    let vocab = tok
        .get("model")
        .and_then(|m| m.get("vocab"))
        .and_then(|v| v.as_object())
        .unwrap_or_else(|| die("tokenizer.json: missing model.vocab"));
    let n = vocab
        .values()
        .filter_map(|v| v.as_u64())
        .max()
        .unwrap_or(0) as usize
        + 1;
    let mut v = vec![String::new(); n];
    for (token, id) in vocab {
        if let Some(i) = id.as_u64() {
            v[i as usize] = token.clone();
        }
    }
    if let Some(a) = tok.get("added_tokens").and_then(|x| x.as_array()) {
        for t in a {
            let content = t.get("content").and_then(|c| c.as_str()).unwrap_or("");
            let id = t.get("id").and_then(|x| x.as_u64());
            if let Some(i) = id {
                if (i as usize) < v.len() {
                    v[i as usize] = content.to_string();
                } else {
                    v.resize(i as usize + 1, String::new());
                    v[i as usize] = content.to_string();
                }
            }
        }
    }
    v
}

fn build_merges(tok: &J) -> Vec<String> {
    tok.get("model")
        .and_then(|m| m.get("merges"))
        .and_then(|v| v.as_array())
        .unwrap_or_else(|| die("tokenizer.json: missing model.merges"))
        .iter()
        .map(|m| match m {
            J::Array(a) => format!(
                "{} {}",
                a[0].as_str().unwrap_or(""),
                a[1].as_str().unwrap_or("")
            ),
            J::String(s) => s.clone(),
            _ => die("tokenizer.json: merge is neither array nor string"),
        })
        .collect()
}

fn build_token_types(tok: &J, n: usize) -> Vec<i32> {
    let mut t = vec![1i32; n];
    if let Some(a) = tok.get("added_tokens").and_then(|x| x.as_array()) {
        for tok in a {
            let id = tok.get("id").and_then(|x| x.as_u64()).unwrap_or(u64::MAX);
            let special = tok.get("special").and_then(|x| x.as_bool()).unwrap_or(false);
            if (id as usize) < n {
                t[id as usize] = if special { 3 } else { 4 };
            }
        }
    }
    t
}
