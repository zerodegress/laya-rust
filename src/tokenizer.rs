use serde_json::Value;
use std::collections::HashMap;

pub struct Tokenizer {
    vocab: HashMap<String, i64>,
    ranks: HashMap<String, i64>,
    added: Vec<(String, i64, bool)>,
    byte_map: HashMap<u8, char>,
    pub cls: i64,
    pub sep: i64,
    pub mask: i64,
    pub pad: i64,
    pub unk: i64,
}

fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut bs: Vec<u32> = Vec::new();
    for b in 33..=126u32 {
        bs.push(b);
    }
    for b in 161..=172u32 {
        bs.push(b);
    }
    for b in 174..=255u32 {
        bs.push(b);
    }
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.into_iter().zip(cs).map(|(b, c)| (b as u8, char::from_u32(c).unwrap())).collect()
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

fn is_number(c: char) -> bool {
    c.is_numeric()
}

fn is_space(c: char) -> bool {
    c.is_whitespace()
}

const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

fn pre_tokenize(text: &str) -> Vec<String> {
    let ch: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < ch.len() {
        let rest: String = ch[i..].iter().take(3).collect();
        let lower = rest.to_lowercase();
        let mut matched = None;
        for c in CONTRACTIONS {
            if lower.starts_with(c) {
                matched = Some(c.chars().count());
                break;
            }
        }
        if let Some(l) = matched {
            out.push(ch[i..i + l].iter().collect());
            i += l;
            continue;
        }
        let sp = ch[i] == ' ';
        let j = if sp { i + 1 } else { i };
        if j < ch.len() && is_letter(ch[j]) {
            let mut k = j;
            while k < ch.len() && is_letter(ch[k]) {
                k += 1;
            }
            out.push(ch[i..k].iter().collect());
            i = k;
            continue;
        }
        if j < ch.len() && is_number(ch[j]) {
            let mut k = j;
            while k < ch.len() && is_number(ch[k]) {
                k += 1;
            }
            out.push(ch[i..k].iter().collect());
            i = k;
            continue;
        }
        if j < ch.len() && !is_space(ch[j]) && !is_letter(ch[j]) && !is_number(ch[j]) {
            let mut k = j;
            while k < ch.len() && !is_space(ch[k]) && !is_letter(ch[k]) && !is_number(ch[k]) {
                k += 1;
            }
            out.push(ch[i..k].iter().collect());
            i = k;
            continue;
        }
        if is_space(ch[i]) {
            let mut k = i;
            while k < ch.len() && is_space(ch[k]) {
                k += 1;
            }
            if k < ch.len() && k - i > 1 {
                k -= 1;
            }
            out.push(ch[i..k].iter().collect());
            i = k;
            continue;
        }
        out.push(ch[i].to_string());
        i += 1;
    }
    out
}

impl Tokenizer {
    pub fn load(path: &str) -> Tokenizer {
        let s = std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("{}: {}", path, e);
            std::process::exit(2);
        });
        Tokenizer::from_json(&s)
    }

    pub fn from_json(text: &str) -> Tokenizer {
        let v: Value = serde_json::from_str(text).unwrap();
        let mut vocab = HashMap::new();
        for (k, id) in v["model"]["vocab"].as_object().unwrap() {
            vocab.insert(k.clone(), id.as_i64().unwrap());
        }
        let mut ranks = HashMap::new();
        for (i, m) in v["model"]["merges"].as_array().unwrap().iter().enumerate() {
            let pair = match m {
                Value::Array(a) => format!("{} {}", a[0].as_str().unwrap(), a[1].as_str().unwrap()),
                Value::String(s) => s.clone(),
                _ => panic!(),
            };
            ranks.insert(pair, i as i64);
        }
        let mut added = Vec::new();
        for a in v["added_tokens"].as_array().unwrap() {
            let id = a["id"].as_i64().unwrap();
            let content = a["content"].as_str().unwrap().to_string();
            let lstrip = a["lstrip"].as_bool().unwrap_or(false);
            added.push((content, id, lstrip));
        }
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let spec = |name: &str| -> i64 {
            v["added_tokens"]
                .as_array()
                .unwrap()
                .iter()
                .find(|a| a["content"].as_str() == Some(name))
                .map(|a| a["id"].as_i64().unwrap())
                .unwrap()
        };
        Tokenizer {
            vocab,
            ranks,
            added,
            byte_map: bytes_to_unicode(),
            cls: spec("[CLS]"),
            sep: spec("[SEP]"),
            mask: spec("[MASK]"),
            pad: spec("[PAD]"),
            unk: spec("[UNK]"),
        }
    }

    fn bpe(&self, word: &str) -> Vec<String> {
        let mut syms: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if syms.len() < 2 {
            return syms;
        }
        loop {
            let mut best: Option<(i64, usize)> = None;
            for i in 0..syms.len() - 1 {
                let key = format!("{} {}", syms[i], syms[i + 1]);
                if let Some(r) = self.ranks.get(&key) {
                    if best.is_none() || *r < best.unwrap().0 {
                        best = Some((*r, i));
                    }
                }
            }
            let (_, at) = match best {
                Some(b) => b,
                None => break,
            };
            let merged = format!("{}{}", syms[at], syms[at + 1]);
            let mut next = Vec::with_capacity(syms.len());
            let mut i = 0;
            while i < syms.len() {
                if i + 1 < syms.len() && syms[i] == syms[at] && syms[i + 1] == syms[at + 1] {
                    next.push(merged.clone());
                    i += 2;
                } else {
                    next.push(syms[i].clone());
                    i += 1;
                }
            }
            syms = next;
            if syms.len() == 1 {
                break;
            }
        }
        syms
    }

    fn encode_chunk(&self, text: &str, out: &mut Vec<i64>) {
        for piece in pre_tokenize(text) {
            let mut mapped = String::new();
            for b in piece.as_bytes() {
                mapped.push(self.byte_map[b]);
            }
            for s in self.bpe(&mapped) {
                out.push(*self.vocab.get(&s).unwrap_or(&self.unk));
            }
        }
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<i64> {
        let ch: Vec<char> = text.chars().collect();
        let mut ids = Vec::new();
        let mut buf = String::new();
        let mut i = 0usize;
        while i < ch.len() {
            let mut hit: Option<(&str, i64, bool, usize)> = None;
            for (content, id, lstrip) in &self.added {
                let n = content.chars().count();
                if n > 0 && i + n <= ch.len() && ch[i..i + n].iter().collect::<String>() == *content {
                    hit = Some((content.as_str(), *id, *lstrip, n));
                    break;
                }
            }
            match hit {
                Some((_, id, lstrip, n)) => {
                    if lstrip {
                        while buf.ends_with(|c: char| c.is_whitespace()) {
                            buf.pop();
                        }
                    }
                    self.encode_chunk(&buf, &mut ids);
                    buf.clear();
                    ids.push(id);
                    i += n;
                }
                None => {
                    buf.push(ch[i]);
                    i += 1;
                }
            }
        }
        self.encode_chunk(&buf, &mut ids);
        if add_special_tokens {
            let mut r = vec![self.cls];
            r.extend(ids);
            r.push(self.sep);
            r
        } else {
            ids
        }
    }
}
