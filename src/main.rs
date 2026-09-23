use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use laya_rust::engine::{BackendId, Engine, EngineInfo, Out, Req};
use laya_rust::systemone::{self, Question};
use laya_rust::tokenizer::Tokenizer;
use laya_rust::weights::Tensors;
use laya_rust::{
    arch, backend, convert, floats, gguf_path, model_path, model_spec, tokenizer_json,
};
use serde_json::{Map, Value, json};
use std::io::Read;

const DEFAULT_MAX_TOKENS: usize = 16384;
const DEFAULT_OUT: &str = "tests/fixtures";
const DEFAULT_ITERS: usize = 50;
const DEFAULT_WARMUP: usize = 3;
const DEFAULT_BITS: i32 = 4;
const DEFAULT_GROUP: i32 = 64;

const MODEL_HELP: &str =
    "model directory or .gguf file (default: $LAYA_MODEL, else models/laya)";
const MODEL_HELP_CONVERT: &str = "checkpoint directory (--to gguf), or model directory / .gguf file (--to mlx); default: $LAYA_MODEL, else models/laya";

const REQUEST_HELP: &str = "\
request (TypeSafe System One shape):
  { \"state\": string|object|array,
    \"model\": \"...\",
    \"questions\": { \"<id>\": { \"type\": \"choice\"|\"score\"|\"noul\",
                               \"instructions\": string|object|array,
                               \"criteria\": ... } } }

  choice criteria: map<option, description|null> (or array of options)
  score  criteria: ordered array of level descriptions (2..10)
  noul   criteria: optional {\"true\": ..., \"false\": ...}

response:
  { \"model\": ..., \"answers\": { \"<id>\": {...} }, \"usage\": {...} }";

const SCENARIOS_HELP: &str = "\
scenarios:
  { \"scenarios\": [ { \"name\": \"...\", \"request\": { ... } }, ... ] }
  a bare array of the same, or a single request object, are also accepted.

request (TypeSafe System One shape):
  { \"state\": string|object|array,
    \"questions\": { \"<id>\": { \"type\": \"choice\"|\"score\"|\"noul\",
                               \"instructions\": string|object|array,
                               \"criteria\": ... } } }";

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendArg {
    Auto,
    Cuda,
    Mlx,
    Cpu,
}

impl BackendArg {
    fn id(self) -> BackendId {
        match self {
            BackendArg::Auto => BackendId::Auto,
            BackendArg::Cuda => BackendId::Cuda,
            BackendArg::Mlx => BackendId::Mlx,
            BackendArg::Cpu => BackendId::Cpu,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ConvertTo {
    Gguf,
    Mlx,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum GroupArg {
    #[value(name = "32")]
    G32,
    #[value(name = "64")]
    G64,
    #[value(name = "128")]
    G128,
}

#[cfg(feature = "mlx")]
impl GroupArg {
    fn value(self) -> i32 {
        match self {
            GroupArg::G32 => 32,
            GroupArg::G64 => 64,
            GroupArg::G128 => 128,
        }
    }
}

#[derive(Parser)]
#[command(
    name = "laya-rust",
    version,
    about = "Laya inference engine: run, benchmark, tokenize, record fixtures and convert weights",
    after_help = REQUEST_HELP
)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    Test(TestArgs),
    Bench(BenchArgs),
    Tokenize(TokenizeArgs),
    Weights(WeightsArgs),
    Golden(GoldenArgs),
    Convert(ConvertArgs),
}

#[derive(Args)]
#[command(
    about = "Run one request and print the System One response",
    after_help = REQUEST_HELP
)]
struct TestArgs {
    #[arg(long, short = 'm', value_name = "PATH", help = MODEL_HELP)]
    model: Option<String>,
    #[arg(long, value_name = "NAME", default_value = "auto", help = "inference backend")]
    backend: BackendArg,
    #[arg(long, help = "mlx only: load mlx/weights.safetensors instead of the dense GGUF values")]
    quantized: bool,
    #[arg(long, short = 'v', help = "timing details on stderr")]
    verbose: bool,
    #[arg(long, value_name = "N", default_value_t = DEFAULT_MAX_TOKENS, help = "padded-token budget per forward")]
    max_tokens: usize,
    #[arg(value_name = "REQUEST", help = "request JSON, '-' for stdin, or omitted for the builtin demo")]
    input: Option<String>,
}

#[derive(Args)]
#[command(
    about = "Measure the steady-state latency distribution",
    after_help = REQUEST_HELP
)]
struct BenchArgs {
    #[arg(long, short = 'm', value_name = "PATH", help = MODEL_HELP)]
    model: Option<String>,
    #[arg(long, value_name = "NAME", default_value = "auto", help = "inference backend")]
    backend: BackendArg,
    #[arg(long, help = "mlx only: load mlx/weights.safetensors instead of the dense GGUF values")]
    quantized: bool,
    #[arg(long, short = 'v', help = "timing details on stderr")]
    verbose: bool,
    #[arg(long, value_name = "N", default_value_t = DEFAULT_MAX_TOKENS, help = "padded-token budget per forward")]
    max_tokens: usize,
    #[arg(long, value_name = "N", default_value_t = DEFAULT_ITERS, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..), help = "timed iterations")]
    iters: usize,
    #[arg(long, value_name = "N", default_value_t = DEFAULT_WARMUP, help = "warmup iterations")]
    warmup: usize,
    #[arg(value_name = "REQUEST", help = "request JSON, '-' for stdin, or omitted for the builtin demo")]
    input: Option<String>,
}

#[derive(Args)]
#[command(
    about = "Tokenize a request and dump the sequences without loading a backend",
    after_help = REQUEST_HELP
)]
struct TokenizeArgs {
    #[arg(long, short = 'm', value_name = "PATH", help = MODEL_HELP)]
    model: Option<String>,
    #[arg(value_name = "REQUEST", help = "request JSON, '-' for stdin, or omitted for the builtin demo")]
    input: Option<String>,
}

#[derive(Args)]
#[command(about = "Dump the tensor table read back from the GGUF")]
struct WeightsArgs {
    #[arg(long, short = 'm', value_name = "PATH", help = MODEL_HELP)]
    model: Option<String>,
}

#[derive(Args)]
#[command(
    about = "Record golden fixtures for the parity tests",
    after_help = SCENARIOS_HELP
)]
struct GoldenArgs {
    #[arg(long, short = 'm', value_name = "PATH", help = MODEL_HELP)]
    model: Option<String>,
    #[arg(long, value_name = "NAME", default_value = "auto", help = "inference backend")]
    backend: BackendArg,
    #[arg(long, help = "mlx only: load mlx/weights.safetensors instead of the dense GGUF values")]
    quantized: bool,
    #[arg(long, value_name = "N", default_value_t = DEFAULT_MAX_TOKENS, help = "padded-token budget per forward")]
    max_tokens: usize,
    #[arg(long, value_name = "DIR", default_value = DEFAULT_OUT, help = "fixture output directory")]
    out: String,
    #[arg(value_name = "SCENARIOS", help = "scenario file path, '-' for stdin, or omitted for stdin")]
    scenarios: Option<String>,
}

#[derive(Args)]
#[command(about = "Write a GGUF from the safetensors checkpoint, or MLX affine weights from the GGUF")]
struct ConvertArgs {
    #[arg(long, short = 'm', value_name = "PATH", help = MODEL_HELP_CONVERT)]
    model: Option<String>,
    #[arg(long, value_name = "NAME", help = "output format")]
    to: ConvertTo,
    #[arg(long, value_name = "PATH", help = "output path (default: the model directory's laya-f16.gguf, laya-f32.gguf with --f32, or mlx/weights.safetensors)")]
    out: Option<String>,
    #[arg(long = "f32", help = "convert --to gguf: widen every tensor to F32 instead of F16")]
    f32_out: bool,
    #[arg(long, value_name = "N", default_value_t = DEFAULT_BITS, value_parser = clap::value_parser!(i32).range(2..=8), help = "convert --to mlx: affine weight bits")]
    bits: i32,
    #[arg(long, value_name = "N", default_value = "64", help = "convert --to mlx: affine group size")]
    group: GroupArg,
}

fn reject_removed(args: &[String]) {
    for a in args {
        let flag = a.split('=').next().unwrap_or(a.as_str());
        let msg = match flag {
            "--dtype" => "--dtype is gone: a GGUF declares its own tensor types. Use --quantized for mlx affine weights.",
            "--weights-json" => "--weights-json is gone: GGUF carries its own tensor table.",
            "--gemm" => "--gemm is gone: the f16 plugin GEMM was slower than cuBLAS and outside the logit tolerance; cuBLAS is now the only path.",
            _ => continue,
        };
        eprintln!("{}", msg);
        std::process::exit(2);
    }
}

fn qdemo() -> Value {
    json!({
        "state": "Help! My payouts have been failing for 3 days.",
        "model": "laya-rust",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Payments, invoicing, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": "Pricing, upgrades, new accounts"
                }
            },
            "is_urgent": {
                "type": "noul",
                "instructions": "Does this convey urgency?"
            },
            "frustration": {
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "criteria": ["Calm", "Frustrated", "Very angry"]
            }
        }
    })
}

struct Item {
    id: String,
    seq: Vec<i64>,
    markers: Vec<i64>,
    qtype: i64,
}

struct Plan {
    questions: Vec<Question>,
    items: Vec<Item>,
}

struct Prepared {
    engine: Box<dyn Engine>,
    info: EngineInfo,
    plan: Plan,
    dir: String,
    tok_ms: f64,
    load_ms: f64,
}

struct Batch {
    n: usize,
    l: usize,
    k: usize,
    ids: Vec<i64>,
    att: Vec<i64>,
    mpos: Vec<i64>,
    mmask: Vec<i64>,
    qtype: Vec<i64>,
}

impl Batch {
    fn req(&self) -> Req<'_> {
        Req {
            n: self.n,
            l: self.l,
            k: self.k,
            ids: &self.ids,
            att: &self.att,
            mpos: &self.mpos,
            mmask: &self.mmask,
            qtype: &self.qtype,
        }
    }
}

struct Executed {
    batch: Batch,
    out: Out,
    rows: Vec<usize>,
}

struct Run {
    forwards: Vec<Executed>,
    tokens: usize,
    answers: Map<String, Value>,
}

struct Load {
    dir: String,
    backend: BackendId,
    quantized: bool,
    bits: i32,
    group: i32,
}

fn resolve_backend(requested: BackendId) -> BackendId {
    if requested == BackendId::Auto {
        backend::auto_id()
    } else {
        requested
    }
}

fn load_engine(load: &Load) -> (Box<dyn Engine>, EngineInfo, f64) {
    let backend_id = resolve_backend(load.backend);
    if load.quantized && backend_id != BackendId::Mlx {
        eprintln!(
            "--quantized is only available on the mlx backend (selected: {})",
            backend_id.name()
        );
        std::process::exit(2);
    }
    let t = std::time::Instant::now();
    let mut spec = model_spec(&load.dir, backend_id, load.quantized);
    spec.quant_bits = load.bits;
    spec.quant_group = load.group;
    let engine = backend::open(&spec);
    let load_ms = t.elapsed().as_secs_f64() * 1000.0;
    let info = engine.info();
    (engine, info, load_ms)
}

fn plan_items(tok: &Tokenizer, state: &Value, questions: &[Question]) -> Plan {
    let mut items = Vec::new();
    for q in questions {
        let (seq, markers) = systemone::build_sequence(tok, state, q);
        let expect = systemone::render_options(q).len();
        if markers.len() != expect {
            eprintln!(
                "question {}: {} options do not fit in head budget ({} markers)",
                q.id,
                expect,
                markers.len()
            );
            std::process::exit(2);
        }
        items.push(Item {
            id: q.id.clone(),
            seq,
            markers,
            qtype: q.qtype,
        });
    }
    Plan {
        questions: questions.to_vec(),
        items,
    }
}

fn prepare(load: &Load, req: &Value) -> Prepared {
    let (state, questions) = systemone::parse_request(req);
    if questions.is_empty() {
        eprintln!("no questions");
        std::process::exit(2);
    }
    let t0 = std::time::Instant::now();
    let tok = Tokenizer::from_json(&tokenizer_json(&load.dir));
    let tok_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (engine, info, load_ms) = load_engine(load);
    let plan = plan_items(&tok, &state, &questions);
    Prepared {
        engine,
        info,
        plan,
        dir: load.dir.clone(),
        tok_ms,
        load_ms,
    }
}

fn group_by_budget(items: &[Item], max_tokens: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by_key(|i| items[*i].seq.len());
    let mut groups = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut cur_max = 0usize;
    for i in order {
        let ln = items[i].seq.len();
        let new_max = cur_max.max(ln);
        if !cur.is_empty() && new_max * (cur.len() + 1) > max_tokens {
            groups.push(std::mem::take(&mut cur));
            cur_max = 0;
        }
        cur_max = cur_max.max(ln);
        cur.push(i);
    }
    if !cur.is_empty() {
        groups.push(cur);
    }
    groups
}

fn build_batch(items: &[Item], idxs: &[usize]) -> Batch {
    let n = idxs.len();
    let l = idxs.iter().map(|i| items[*i].seq.len()).max().unwrap();
    let k = idxs.iter().map(|i| items[*i].markers.len()).max().unwrap();
    let pad = 0i64;
    let mut ids = vec![pad; n * l];
    let mut att = vec![0i64; n * l];
    let mut mpos = vec![0i64; n * k];
    let mut mmask = vec![0i64; n * k];
    let mut qtype = vec![0i64; n];
    for (r, &i) in idxs.iter().enumerate() {
        let it = &items[i];
        ids[r * l..r * l + it.seq.len()].copy_from_slice(&it.seq);
        att[r * l..r * l + it.seq.len()].fill(1);
        for (j, m) in it.markers.iter().enumerate() {
            mpos[r * k + j] = *m;
            mmask[r * k + j] = 1;
        }
        qtype[r] = it.qtype;
    }
    Batch {
        n,
        l,
        k,
        ids,
        att,
        mpos,
        mmask,
        qtype,
    }
}

fn score_row(
    out: &Out,
    row: usize,
    k: usize,
    markers: usize,
    qtype: i64,
    cal: &systemone::Calibration,
) -> (Vec<f32>, f32) {
    let raw = &out.logits[row * k..row * k + markers];
    let t = systemone::temperature(qtype, markers, cal);
    let probs = systemone::softmax_t(raw, t);
    let a = &out.act[row * 2..row * 2 + 2];
    let m = a.iter().cloned().fold(f32::MIN, f32::max);
    let e0 = (a[0] - m).exp();
    let e1 = (a[1] - m).exp();
    (probs, e0 / (e0 + e1))
}

fn run_all(engine: &mut dyn Engine, plan: &Plan, max_tokens: usize) -> Run {
    let cal = engine.info().calibration;
    let groups = group_by_budget(&plan.items, max_tokens);
    let mut answers = Map::new();
    let mut tokens = 0usize;
    let mut forwards = Vec::new();
    for g in &groups {
        tokens += g.iter().map(|i| plan.items[*i].seq.len()).sum::<usize>();
        let batch = build_batch(&plan.items, g);
        let out = engine.forward(&batch.req());
        for (row, &i) in g.iter().enumerate() {
            let it = &plan.items[i];
            let (probs, act_p) = score_row(&out, row, batch.k, it.markers.len(), it.qtype, &cal);
            answers.insert(
                it.id.clone(),
                systemone::answer(&plan.questions[i], &probs, act_p),
            );
        }
        forwards.push(Executed {
            batch,
            out,
            rows: g.clone(),
        });
    }
    Run {
        forwards,
        tokens,
        answers,
    }
}

fn response(run: &Run) -> Value {
    json!({
        "model": "laya-rust",
        "answers": run.answers,
        "usage": { "input_tokens": run.tokens, "output_tokens": 0 }
    })
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    reject_removed(&argv);
    match Cli::parse().cmd {
        Some(Cmd::Test(a)) => cmd_test(a),
        Some(Cmd::Bench(a)) => cmd_bench(a),
        Some(Cmd::Tokenize(a)) => cmd_tokenize(a),
        Some(Cmd::Weights(a)) => cmd_weights(a),
        Some(Cmd::Golden(a)) => cmd_golden(a),
        Some(Cmd::Convert(a)) => cmd_convert(a),
        None => {
            let _ = Cli::command().print_help();
        }
    }
}

fn read_text(input: Option<&str>) -> String {
    match input {
        Some("-") | None => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s).unwrap();
            s
        }
        Some(s) => s.to_string(),
    }
}

fn read_req(input: Option<&str>) -> Value {
    let raw = read_text(input);
    if raw.trim().is_empty() {
        return qdemo();
    }
    match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("invalid json: {}", e);
            std::process::exit(2);
        }
    }
}

fn cmd_test(a: TestArgs) {
    let load = Load {
        dir: model_path(a.model.as_deref()),
        backend: a.backend.id(),
        quantized: a.quantized,
        bits: DEFAULT_BITS,
        group: DEFAULT_GROUP,
    };
    let req = read_req(a.input.as_deref());
    let mut p = prepare(&load, &req);
    let n = p.plan.items.len();
    let l = p.plan.items.iter().map(|i| i.seq.len()).max().unwrap();
    let k = p.plan.items.iter().map(|i| i.markers.len()).max().unwrap();

    let t = std::time::Instant::now();
    let mut run = run_all(&mut p.engine, &p.plan, a.max_tokens);
    let first_ms = t.elapsed().as_secs_f64() * 1000.0;
    let mut times = Vec::new();
    for _ in 0..5 {
        let t = std::time::Instant::now();
        run = run_all(&mut p.engine, &p.plan, a.max_tokens);
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let fwd_ms = times.iter().cloned().fold(f64::MAX, f64::min);
    let avg_ms = times.iter().sum::<f64>() / times.len() as f64;

    if a.verbose {
        eprintln!(
            "model={} backend={} device={:?} dtype={} layers={} questions={} seq_len={} markers={} forwards={} tokenizer={:.1}ms load={:.0}ms",
            p.dir,
            p.info.id.name(),
            p.info.device,
            p.info.dtype.name(),
            p.info.layers,
            n,
            l,
            k,
            run.forwards.len(),
            p.tok_ms,
            p.load_ms
        );
        eprintln!(
            "first={:.2}ms  per-request={:.2}ms (avg {:.2}ms, {:.1} q/s)",
            first_ms,
            fwd_ms,
            avg_ms,
            n as f64 * 1000.0 / fwd_ms
        );
    }
    println!("{}", serde_json::to_string(&response(&run)).unwrap());
}

fn cmd_bench(a: BenchArgs) {
    let load = Load {
        dir: model_path(a.model.as_deref()),
        backend: a.backend.id(),
        quantized: a.quantized,
        bits: DEFAULT_BITS,
        group: DEFAULT_GROUP,
    };
    let req = read_req(a.input.as_deref());
    let mut p = prepare(&load, &req);
    let n = p.plan.items.len();
    let l = p.plan.items.iter().map(|i| i.seq.len()).max().unwrap();
    let k = p.plan.items.iter().map(|i| i.markers.len()).max().unwrap();
    let groups = group_by_budget(&p.plan.items, a.max_tokens).len();

    for _ in 0..a.warmup {
        run_all(&mut p.engine, &p.plan, a.max_tokens);
    }
    let mut times = Vec::with_capacity(a.iters);
    let mut run = run_all(&mut p.engine, &p.plan, a.max_tokens);
    for _ in 0..a.iters {
        let t = std::time::Instant::now();
        run = run_all(&mut p.engine, &p.plan, a.max_tokens);
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let mut sorted = times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = sorted[0];
    let p50 = pct(&sorted, 50.0);
    let p90 = pct(&sorted, 90.0);
    let p99 = pct(&sorted, 99.0);
    let max = sorted[sorted.len() - 1];
    let avg = sorted.iter().sum::<f64>() / sorted.len() as f64;

    let mut out = response(&run);
    out["latency_ms"] = json!({
        "min": min, "p50": p50, "p90": p90, "p99": p99, "max": max, "mean": avg
    });
    out["batch"] = json!({
        "questions": n, "seq_len": l, "markers": k, "forwards": groups,
        "iters": a.iters, "warmup": a.warmup,
        "load_ms": p.load_ms, "tokenizer_ms": p.tok_ms,
        "per_question_ms": avg / n as f64,
        "questions_per_s": n as f64 * 1000.0 / avg
    });
    if a.verbose {
        eprintln!(
            "model={} backend={} device={:?} dtype={} questions={} seq_len={} markers={} forwards={} load={:.0}ms",
            p.dir,
            p.info.id.name(),
            p.info.device,
            p.info.dtype.name(),
            n,
            l,
            k,
            groups,
            p.load_ms
        );
        eprintln!(
            "latency ms: min={:.3} p50={:.3} p90={:.3} p99={:.3} max={:.3} mean={:.3}",
            min, p50, p90, p99, max, avg
        );
        eprintln!(
            "{:.2} ms/request, {:.2} ms/question, {:.1} questions/s",
            avg,
            avg / n as f64,
            n as f64 * 1000.0 / avg
        );
    }
    println!("{}", serde_json::to_string(&out).unwrap());
}

fn cmd_weights(a: WeightsArgs) {
    let t = Tensors::open(&gguf_path(&model_path(a.model.as_deref())));
    let d = t.dump(&arch::weight_names());
    println!("{}", serde_json::to_string_pretty(&d).unwrap());
}

fn cmd_convert(a: ConvertArgs) {
    match a.to {
        ConvertTo::Gguf => {
            let model = model_path(a.model.as_deref());
            if std::path::Path::new(&model).is_file() {
                eprintln!(
                    "convert --to gguf reads a checkpoint directory, not a gguf file: {}",
                    model
                );
                std::process::exit(2);
            }
            let out = match &a.out {
                Some(o) => o.clone(),
                None if a.f32_out => format!("{}/laya-f32.gguf", model),
                None => gguf_path(&model),
            };
            let t = std::time::Instant::now();
            let s = convert::to_gguf(&model, &out, !a.f32_out);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "out": out,
                    "format": "gguf",
                    "dtype": if s.f16 { "f16" } else { "f32" },
                    "tensors": s.tensors,
                    "copied_verbatim": s.copied,
                    "converted": s.widened,
                    "tokenizer_tokens": s.tokenizer_tokens,
                    "tokenizer_merges": s.tokenizer_merges,
                    "bytes": s.bytes,
                    "convert_ms": ms,
                }))
                .unwrap()
            );
        }
        ConvertTo::Mlx => cmd_convert_mlx(&a),
    }
}

#[cfg(feature = "mlx")]
fn cmd_convert_mlx(a: &ConvertArgs) {
    let gguf = gguf_path(&model_path(a.model.as_deref()));
    let out = a.out.clone().unwrap_or_else(|| {
        let dir = std::path::Path::new(&gguf)
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        format!("{}/mlx/weights.safetensors", dir.display())
    });
    let t = std::time::Instant::now();
    let s = backend::mlx::convert(&gguf, None, None, a.bits, a.group.value(), &out);
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "out": out,
            "format": "mlx-affine",
            "bits": a.bits,
            "group": a.group.value(),
            "tensors": s.names,
            "quantized_tensors": s.quantized,
            "params": s.params,
            "quantized_params": s.quant_params,
            "kept_f32_tensors": s.names - s.quantized,
            "bytes": s.bytes,
            "convert_ms": ms,
        }))
        .unwrap()
    );
}

#[cfg(not(feature = "mlx"))]
fn cmd_convert_mlx(_a: &ConvertArgs) {
    eprintln!("convert --to mlx requires a build with --features mlx");
    std::process::exit(2);
}

fn cmd_tokenize(a: TokenizeArgs) {
    let req = read_req(a.input.as_deref());
    let (state, questions) = systemone::parse_request(&req);
    let model = model_path(a.model.as_deref());
    let tok = Tokenizer::from_json(&tokenizer_json(&model));
    let mut out = Map::new();
    for q in &questions {
        let opts = systemone::render_options(q);
        let (ids, markers) = systemone::build_sequence(&tok, &state, q);
        out.insert(
            q.id.clone(),
            json!({
                "type": q.ty, "qtype": q.qtype, "options": opts,
                "ids": ids, "markers": markers,
                "seq_len": ids.len()
            }),
        );
    }
    println!("{}", serde_json::to_string(&Value::Object(out)).unwrap());
}

struct Scenario {
    name: String,
    request: Value,
    state: Value,
    questions: Vec<Question>,
}

fn read_scenarios_arg(input: Option<&str>) -> String {
    match input {
        None | Some("-") => read_text(None),
        Some(path) => std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("{}: {}", path, e);
            std::process::exit(2);
        }),
    }
}

fn parse_scenarios(raw: &str) -> Vec<Scenario> {
    let v: Value = serde_json::from_str(raw).unwrap_or_else(|e| {
        eprintln!("invalid json: {}", e);
        std::process::exit(2);
    });
    let entries: Vec<(String, Value)> = match &v {
        Value::Object(o) if o.contains_key("scenarios") => o["scenarios"]
            .as_array()
            .unwrap_or_else(|| {
                eprintln!("\"scenarios\" must be an array");
                std::process::exit(2);
            })
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let name = s
                    .get("name")
                    .and_then(|x| x.as_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("scenario{}", i));
                let req = s.get("request").cloned().unwrap_or_else(|| s.clone());
                (name, req)
            })
            .collect(),
        Value::Array(a) => a
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let name = s
                    .get("name")
                    .and_then(|x| x.as_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("scenario{}", i));
                let req = s.get("request").cloned().unwrap_or_else(|| s.clone());
                (name, req)
            })
            .collect(),
        Value::Object(_) => vec![("default".to_string(), v.clone())],
        _ => {
            eprintln!("scenarios must be an object with \"scenarios\", an array, or a request object");
            std::process::exit(2);
        }
    };
    let mut out = Vec::new();
    for (name, request) in entries {
        let (state, questions) = systemone::parse_request(&request);
        if questions.is_empty() {
            eprintln!("scenario {}: no questions", name);
            std::process::exit(2);
        }
        out.push(Scenario {
            name,
            request,
            state,
            questions,
        });
    }
    out
}

fn cmd_golden(a: GoldenArgs) {
    let load = Load {
        dir: model_path(a.model.as_deref()),
        backend: a.backend.id(),
        quantized: a.quantized,
        bits: DEFAULT_BITS,
        group: DEFAULT_GROUP,
    };
    let scenarios = parse_scenarios(&read_scenarios_arg(a.scenarios.as_deref()));
    if scenarios.is_empty() {
        eprintln!("no scenarios");
        std::process::exit(2);
    }
    let tok = Tokenizer::from_json(&tokenizer_json(&load.dir));
    let (mut engine, info, load_ms) = load_engine(&load);
    std::fs::create_dir_all(&a.out)
        .unwrap_or_else(|e| {
            eprintln!("{}: {}", a.out, e);
            std::process::exit(2);
        });
    eprintln!(
        "golden: backend={} device={:?} dtype={} load={:.0}ms out={} scenarios={}",
        info.id.name(),
        info.device,
        info.dtype.name(),
        load_ms,
        a.out,
        scenarios.len()
    );
    for sc in &scenarios {
        let plan = plan_items(&tok, &sc.state, &sc.questions);
        let run = run_all(&mut engine, &plan, a.max_tokens);
        let forwards: Vec<Value> = run
            .forwards
            .iter()
            .map(|ex| {
                let rows: Vec<Value> = ex
                    .rows
                    .iter()
                    .map(|i| {
                        json!({
                            "id": plan.items[*i].id,
                            "markers": plan.items[*i].markers.len(),
                            "qtype": plan.items[*i].qtype,
                        })
                    })
                    .collect();
                json!({
                    "n": ex.batch.n, "l": ex.batch.l, "k": ex.batch.k,
                    "ids": ex.batch.ids,
                    "att": ex.batch.att,
                    "mpos": ex.batch.mpos,
                    "mmask": ex.batch.mmask,
                    "qtype": ex.batch.qtype,
                    "rows": rows,
                    "logits": floats(&ex.out.logits),
                    "act": floats(&ex.out.act),
                })
            })
            .collect();
        let doc = json!({
            "name": sc.name,
            "backend": info.id.name(),
            "dtype": info.dtype.name(),
            "request": sc.request,
            "forwards": forwards,
            "answers": Value::Object(run.answers),
        });
        let path = format!("{}/{}.json", a.out, sc.name);
        std::fs::write(&path, format!("{}\n", serde_json::to_string_pretty(&doc).unwrap()))
            .unwrap_or_else(|e| {
                eprintln!("{}: {}", path, e);
                std::process::exit(2);
            });
        eprintln!(
            "  wrote {} ({} forward(s), {} marker(s))",
            path,
            run.forwards.len(),
            plan.items.iter().map(|i| i.markers.len()).sum::<usize>()
        );
    }
}
