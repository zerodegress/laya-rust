# AGENTS.md

Agent-facing notes for `laya-rust`: the hard rules, and the knowledge a developer or a coding agent needs while working in this repository. [README.md](README.md) is the brief human-facing entry point — what the engine is, how to build it, how to run it — and is deliberately kept short. Nothing is duplicated between the two files; this one carries the detail.

## Hard rules

1. **Never write comments.** No comments in Rust, TOML, shell, CUDA C kernels, or any other file. Code must carry its own meaning; when a decision needs prose it belongs in this file, not next to the code. `src/` and `tests/` currently contain zero comments — keep it that way. The feature note in `Cargo.toml` is grandfathered; do not add more, and delete a comment rather than update it when you touch its block.
2. **`README.md` stays brief; this file carries the detail.** The README is a short usage document for a human who wants to build and run the engine — resist growing it, and put depth here instead. Do not create extra documents (no new `*.md`, no `docs/` tree, no design notes, no changelog); add prose to an existing section rather than starting a new file.
3. **No new dependencies without explicit human approval.** Even a dependency that looks like a large, obvious win requires asking first. Features and dependencies are part of the build contract (exactly one backend per build), and `Cargo.lock` is committed.
4. **No model downloads without explicit human approval.** Weights are not in the repository (`/models` is git-ignored). If a test or benchmark needs a model that is missing, ask the human before fetching anything.

## Build and test

One backend per build; a build with no backend feature is a compile error that names the features (`src/lib.rs`). `cuda` is the default feature. The package is `laya-rust` (library crate `laya_rust`) and the binary it builds is `laya`, which is the name every example below uses.

```bash
cargo build --release                                        # CUDA (default)
cargo build --release --no-default-features --features cpu    # CPU reference backend
cargo build --release --no-default-features --features mlx    # Apple Silicon only

cargo test --release                                          # cuda  (regression vs fixtures)
cargo test --release --no-default-features --features cpu      # cpu   (parity vs fixtures)
cargo test --release --no-default-features --features mlx      # mlx   (parity vs fixtures)
```

`tests/ops.rs` needs no weights and runs anywhere. `tests/gguf.rs` likewise. `tests/golden.rs` skips itself (loudly, naming `LAYA_MODEL_DIR`) unless a model holding the GGUF is present — `LAYA_MODEL_DIR` (a directory, or a `.gguf` file path), or `models/laya`. `LAYA_TEST_BACKEND` forces the backend it uses, which is how the parity check runs in a build that has both compiled (`--features cuda,cpu`). Regenerate fixtures on a machine with a GPU and the weights:

```bash
laya golden --max-tokens 300 --out tests/fixtures tests/scenarios.json
```

`tests/ops.rs` runs the same 18 assertions once per compiled backend (`cpu::…` and `mlx::…`) from a single macro body, so a new backend inherits the whole check set by implementing `ops::Backend` and adding one `suite!` line.

## CLI and environment

`src/main.rs` is a thin `clap` front-end over the library; every flag is scoped and typed per subcommand.

| command | role |
|---|---|
| `laya test [-v] [REQUEST]` | run one request (batched into as few forwards as `--max-tokens` allows) and print the System One response |
| `laya bench [--warmup N] [--iters N] [REQUEST]` | steady-state latency distribution |
| `laya tokenize [REQUEST]` | dump the tokenized sequences without loading a backend |
| `laya weights` | dump the tensor table read back from the GGUF |
| `laya golden [--max-tokens N] --out DIR SCENARIOS` | record the parity fixtures |
| `laya convert --to gguf\|mlx [--out PATH] [--f32] [--bits N] [--group N]` | checkpoint → GGUF, or GGUF → MLX affine weights |
| `laya serve [--addr ADDR] [--endpoint PATH] [--queue N] [--connections N]` | long-running HTTP service: one `POST` endpoint taking the System One request shape |

`REQUEST` is inline JSON, `-` for stdin, or omitted for a builtin demo — except on `golden`, where the positional is a *path* to a scenario file. `--model PATH` / `-m` is a model *directory* (the GGUF is then `laya-f16.gguf` inside it) or a `.gguf` *file*, and it defaults to `$LAYA_MODEL`, else `models/laya`; `--backend auto|cuda|mlx|cpu` defaults to `auto`, the first backend compiled in the order cuda, mlx, cpu; `--quantized` is mlx-only and resolves to `mlx/weights.safetensors` next to the GGUF; `--max-tokens N` defaults to 16384; `-v` / `--verbose` adds timing detail on stderr. `--to` is required on `convert` (`--to gguf` reads the checkpoint *directory* and refuses a `.gguf` path — the store is the input there; `--to mlx` takes either form), and `--bits` / `--group` are validated by the parser (2..8, and 32 / 64 / 128) rather than by a panic or a silent cast. `--help`, `help <command>` and `-V` are `clap`'s.

Flags that described the removed ONNX sidecar (`--dtype`, `--weights-json`) and the removed plugin GEMM (`--gemm`) are rejected with an explanation rather than silently ignored, because accepting them would advertise a capability this build does not have.

Environment: `CUDA_PATH` / `CUDA_HOME` override the NVRTC include path (default `/usr/local/cuda`), `LAYA_CPU_THREADS` sets the CPU backend's thread count (default: available parallelism), `LAYA_MODEL` supplies the default for `--model` everywhere, and `LAYA_MODEL_DIR` / `LAYA_TEST_BACKEND` are read by `tests/golden.rs`.

## Serving

`laya serve` puts the same request path behind HTTP, for a process that stays warm. It exists because `laya test` re-does per-request work a service must pay once: `prepare` re-reads the GGUF and re-parses the embedded `tokenizer.huggingface.json`, which is ~34 MB of JSON. Measured, that parse costs **104.6 ms** on the CPU backend (against a 3191 ms load) and **93.7–109.0 ms** on CUDA (against a 3073–3162 ms load) — while the CUDA forward it feeds is **20.9 ms**. Re-parsing per request would therefore have cost ~4.5x the inference it exists to serve, so hoisting both to startup is worth far more than the HTTP layer costs.

The HTTP layer is `tiny_http`: blocking, thread-per-connection, no async runtime. That is not only a size argument (it adds `tiny_http`, `ascii`, `chunked_transfer` and `httpdate`, and `log` was already in the tree). `engine::Engine::forward` takes `&mut self` and no backend is `Sync` — CUDA holds a context, MLX a thread-local stream — so inference is single-threaded by construction. An async runtime would add a scheduler whose only job is to hop back off again, and the CPU backend's per-matmul `std::thread::scope` already saturates every core.

One `laya-infer` thread opens the tokenizer and the engine, then owns them for the life of the process. **The engine is constructed on that thread and never moved to it**: MLX's stream is thread-local and CUDA's context is created on first use, so the thread that builds the engine has to be the thread that runs it. Connection threads (default 4) hand over `(request, reply channel)` on a bounded `sync_channel` and block on the reply, so exactly one forward runs at a time and the queue is the backpressure — a full queue answers `503` rather than buffering without limit. Nine warm CUDA requests served end-to-end at 21.3–22.7 ms, against the CLI's own 20.9 ms forward: the HTTP layer costs about a millisecond. Three concurrent CPU requests all answered `200`, serialized at 1.43 s / 3.33 s / 5.50 s with byte-identical bodies.

The endpoint is `POST --endpoint` (default `/systemone`). The body is the shape under **Request shape (System One)** and the reply is the document `test` prints, byte-identical to it but for the trailing newline `println!` adds. `404` unknown path, `405` not POST, `413` body over 8 MiB, `400` unparseable JSON, `422` a request `systemone::parse_request` rejects, `500` a forward that panicked. The last two are why `systemone::parse_request` and `plan_items` return `Result` instead of calling `process::exit`: a malformed request must not take the service down, and the CLI reproduces the old behaviour by exiting `2` through `parse_request_or_exit` / `plan_items_or_exit`. A panic inside a forward is caught per request with `catch_unwind`, so one bad input cannot kill the inference thread and take the whole service with it — and a failure to load the model, which the backends still report by panicking (see `src/backend/cuda/mod.rs`), is caught by the same seam and turns into one line rather than a silent hang.

Shutdown is the default signal behaviour, deliberately: there is no signal handler and no `ctrlc` dependency, because responses are per-request and the weights are read-only, so there is nothing to flush and `SIGINT` / `SIGTERM` loses nothing. Startup detail is reported once under `--verbose`, and there is no health endpoint — the requirement is one endpoint.

## Model

The weights are the `safetensors` checkpoint of `convaiinnovations/laya`: a **ModernBERT-large** backbone (28 encoder layers) plus a decision head (2 transformer layers, an option-marker scorer and an act/escalate head), 421M parameters, non-autoregressive. It never generates text.

The engine's weight format is **GGUF** and it is the only one. `convert --to gguf` runs once against the checkpoint; afterwards the engine reads only that GGUF, at whatever path `--model` / `$LAYA_MODEL` names — `{model dir}/laya-f16.gguf` for a directory, or the file itself. The ONNX export the CUDA path was originally built against is gone, along with `weights.json` and the `--dtype` file table.

Architecture constants (`src/arch.rs`):

| Symbol | Value | Meaning |
|---|---|---|
| `H` | 1024 | hidden size |
| `NH` | 16 | attention heads |
| `HD` | 64 | head dim (`H == NH * HD`) |
| `NLAYER` | 28 | encoder layers |
| `INTER` | 2624 | half of the gated MLP width (gate/up is `2 * INTER`) |
| `NVOCAB` | 50368 | vocab size |
| `SCALE` | 0.35355338 | attention scale, `1 / (2*sqrt(2))` |
| `EPS` | 1e-5 | layer-norm epsilon |
| `WINDOW` | 64 | sliding-window radius (`local_attention` is `2 * WINDOW` = 128) |
| `HEAD_LAYERS` | 2 | decision-head transformer layers |
| `HEAD_INTER` | 4096 | decision-head MLP width |
| `ACT_HIDDEN` | 256 | act/escalate head hidden width |
| `ACT_INPUT` | `H + 4` = 1028 | act-head input: first token plus 4 marker features |
| `QTYPES` | 3 | question types |
| `GLOBAL_EVERY` | 3 | every third encoder layer is global (`layer_types` in `encoder/config.json` spells this out) |

Tensor names are the checkpoint's own (`encoder.layers.0.attn.Wqkv.weight`, `head.layers.*`, `scorer.*`, `act_head.*`); the roles live in `arch.rs` and the GGUF stores those names verbatim, so no name translation happens at load. The upstream exporter's opaque `val_*` names are not used anywhere.

Conventions a new backend must reproduce exactly. Getting any of these wrong produces plausible-looking but wrong probabilities, which is why they are pinned by tests:

* **Attention scale lives inside the qkv split.** `q` and `k` are multiplied by `SCALE` when they are split out of the fused `qkv` projection; `v` is not scaled. The attention op therefore must *not* scale again.
* **RoPE is rotate-half (GPT-NeoX style), table-driven.** For head dim `d` with `half = HD/2`, the partner is `d + half` (or `d - half`), negated when `d < half`: `q'[d] = q[d]*cos - q[d+half]*sin` for `d < half`, and `q'[d] = q[d]*cos + q[d-half]*sin` otherwise. Frequencies are `d % half`, and they come from the two tables stored in the GGUF as `rope.freq_full` and `rope.freq_sliding`, not from a computed schedule at load. Tables are indexed by absolute position: `angle = position * freq[j]`.
* **Encoder layers alternate.** Layer `i` uses the global table and the unwindowed mask when `i % 3 == 0`, otherwise the windowed table and a `+/-64` sliding-window mask (`arch::is_global_layer`).
* **The mask is additive and shared.** It is built once per forward as `[n, l, l]` with `0.0` to keep and `-3.4e38` to drop, from the padding mask (`att == 0` drops the *key*) plus the optional window. The head layers use the unwindowed mask. It is added to the scores *before* the softmax.
* **Layer 0 seeds the residual stream** from its own normed input (`x = norm(x) + attn(...)`), later layers accumulate (`x += ...`).
* **Marker scoring.** One score per option marker. Masked markers are replaced by `-10000.0` in the emitted `logits`, but the softmax that produces the action-head features runs over *all* `k` marker columns.
* **Action-head features** are `[top1, top1 - top2, entropy / log(n), n / 255]` of that marker softmax, where `n = max(valid_markers, 2)`. They are concatenated with the first token of each sequence.
* **Calibration** happens outside the model: per-question-type temperatures, refined by option count (`src/systemone.rs`). The backend returns raw marker scores and raw act logits; `systemone` applies the temperature, the softmax, and the answer shaping.

## Weights (GGUF)

`src/gguf.rs` is a dependency-free GGUF v3 reader and writer: header, all 13 metadata value types (arrays nest, element types must agree), the tensor-info table, and the 32-byte alignment rule. It validates as it writes (keys must be ASCII `lower_snake_case` segments joined by `.`, tensor names ASCII and at most 64 bytes, unique, at most 4 dimensions, byte length consistent with shape and type) and as it reads (offsets aligned, tensor data inside the file, array lengths plausible for the remaining file size, so a malformed file cannot trigger an unbounded allocation).

`convert --to gguf` reads the checkpoint's `model.safetensors` directly — its own reader in `src/weights.rs`, no `safetensors` dependency — plus `encoder/config.json` for the hyper-parameters, `rl_agent_config.json` for the calibration, and `tokenizer/tokenizer.json` for the vocabulary. It writes 207 tensors:

* **205 copied byte-for-byte** from the checkpoint, so the GGUF carries exactly the published weights. `F16` stays `F16`; `--f32` widens everything instead.
* **2 recomputed**: `rope.freq_full` and `rope.freq_sliding`, the RoPE frequency tables the checkpoint does not contain. They are `1 / theta^(2j/HD)` evaluated in `f32` with `theta` from `config.json`'s `rope_parameters` (160000 full-attention, 10000 sliding). **These two must be `F32` regardless of `--f16`**: rounding the tables to `f16` moves `0.7498942` to `0.75`, an angle error of 0.08 rad at position 512, which is enough to fail the golden fixtures. They are 32 values each.

The file is self-describing: `general.*` (architecture `laya`, name, author, license, repo, description, `file_type`, alignment), `laya.*` for every hyper-parameter in the constants table above (including `rope_theta.full`/`.sliding`, `local_attention`, `max_len`, `head_max_len`, `max_prefixes`), the calibration (`laya.temperature.0..2` plus `laya.temperature_by_options` as a JSON string, because the option-count keys contain `:` and `-` and GGUF metadata keys may not), and the tokenizer (`tokenizer.ggml.*` including `mask_token_id` 50284 and `cls_token_id` 50281, **and** `tokenizer.huggingface.json` carrying the whole file verbatim, which is what the runtime actually reads — the ggml keys cannot express `[MASK]`'s `lstrip`).

**Calibration does not come from the checkpoint.** `rl_common.py` registers `self.register_buffer("temperature", torch.ones(3))` and notes it is "fitted post-hoc in evaluate.py", so the `temperature` tensor in every checkpoint is a dummy and the fitted values live only in `rl_agent_config.json`. The converter takes them from there and the loader hands them to `systemone` through `EngineInfo.calibration`; they used to be `pub const`s in `src/systemone.rs` fixed to the base model, which silently applied the wrong temperatures to a differently calibrated checkpoint. All three published checkpoints differ: base `[1.637, 1.251, 1.983]` with six option-count refinements, `typed-decisions` `[1.015, 1.037, 1.058]`, `multilingual` `[1, 1, 1]` with none. Only the base is loadable — `multilingual` is a different architecture (22 layers, hidden 768, intermediate 1152, vocab 256000) that these compile-time constants cannot express, which is the case for the loader trusting metadata over `arch.rs`.

### The layout fix-up

GGML stores `ne[0]` contiguous, and `Dimension[0]` in the file is that axis, so the GGUF dimension array is the *reverse* of the framework shape. The checkpoint is PyTorch layout (`nn.Linear` weight is `[out, in]` with `in` contiguous), which is exactly ggml's canonical layout — so the file stores the checkpoint's bytes untouched and the written dimensions are `reversed(shape)`.

The graph, however, wants two layouts, and it wants them for a reason that predates GGUF: `gemm_nt` reads `w[kx*n + ni]` (`[in, out]`, `out` contiguous) for `matmul`, and `gemm_t` reads `w[ni*kk + kx]` (`[out, in]`, `in` contiguous) for `matmul_t`. Canonical storage matches `matmul_t` and the embedding tables directly, so those load as-is. It is the *transpose* of what `matmul` needs, so `Tensors::f32` physically transposes exactly those weights on the way in, and `Tensors::dims` reports the swapped shape so `f32_shaped`'s product assertion holds.

`arch::transpose_names()` is that set, derived from the same tables `graph.rs` indexes rather than from shape heuristics: 120 tensors — `WQKV`, `WO_ATTN`, `WI` and `WO_MLP` for all 28 layers, the head's `in_proj_weight`, `linear1` and `linear2`, and the scorer's hidden and output projections. **It must be role-driven, not shape-driven.** `WO_ATTN`, `scorer.1.weight` and the head's `out_proj` are square, so a `dims[0] != dims[1]` test silently skips them and produces a wrong-but-plausible model. The old `weights.json` heuristic had exactly that hole.

Loading cross-checks rather than trusts: every hyper-parameter in the file must equal the constant `arch.rs` pins (a mismatch exits with both numbers), and every tensor is checked against its expected stored shape and against `F32`/`F16`/`BF16`. This is the "write like the metadata is the truth, read like `arch.rs` is" compromise: the file is honest for any other consumer, and a file this build cannot honour fails loudly at load instead of producing plausible-looking wrong probabilities.

## Op contract

If you are implementing a backend, these are the tensors `graph::forward` will hand you. `rows = n * l` (batch times padded sequence length).

| op | inputs | output |
|---|---|---|
| `embedding` | table `[vocab, H]`, `ids: &[i64]` | `[rows, H]` |
| `layernorm` | `x [rows, H]`, `w [H]`, optional `b [H]` | `[rows, H]` |
| `matmul` | `x [rows, in]`, `w [in, out]` | `[rows, out]` |
| `matmul_t` | `x [rows, in]`, `w [out, in]` | `[rows, out]` |
| `split_qkv_rope` | `qkv [rows, 3H]`, optional `(cos, sin)` each `[l, HD/2]` | `q`, `k`, `v` each `[n, NH, l, HD]` |
| `attention` | `q`,`k`,`v` `[n, NH, l, HD]`, mask `[n, l, l]` | `[rows, H]` |
| `window_mask` | `att: &[i64]` | `[n, l, l]` additive |
| `add`, `add_bias`, `gelu`, `relu` | `x` | same shape |
| `gelu_mul` | `x [rows, 2*INTER]` | `[rows, INTER]` |
| `add_type` | `x [rows, H]`, type embeddings `[3, H]`, `qtype: &[i64]` | `[rows, H]` |
| `gather_markers` | `x [rows, H]`, `mpos: &[i64]` | `[n*k, H]` |
| `post` | scores `[n*k]`, `mmask: &[i64]` | `logits [n*k]`, `feats [n, 4]` |
| `act_input` | `x [rows, H]`, `feats [n, 4]` | `[n, H+4]` |

`post` returns the *masked raw* score per marker (not a softmax): the graph's consumer applies the temperature and softmax. `matmul` vs `matmul_t` is not a detail you can normalise away: the graph keeps two weight layouts on purpose (see **Weights (GGUF)**), and the fused CUDA path uses `mm` for one and `linear` for the other.

The trait also carries a `TapSink` hook: the shared graph can emit named intermediate activations (`embed`, `enc.{i}`, `final`, `head.{i}`, `logits`, `act`). Nothing supplies a sink in normal operation, and supplying one materialises tensors, so it is a diagnostic-only path — but it is how a new backend is localised: run the reference backend and the new one with the same sink and find the first stage that diverges.

## Precision policy

Activations are `f32`. Weight precision is the backend's own business: every op takes and returns `f32` activations, so a backend that wants to compute in `f16` casts internally. Weight *storage* is a separate axis, declared per tensor through the `WeightFormat` protocol (`weight_format` / `upload_f32` / `upload_f16` / `upload_quant` / `resident`), with `graph::load` dispatching on the declaration and every method defaulting to a plain f32 upload (see Quantisation under **Backends**).

The CPU backend stores `f32` and widens only where it is cheap — layer-norm mean/variance and the `erf` inside GELU accumulate in `f64`. It is therefore slightly *more* accurate than the fused CUDA path rather than differently wrong. Matrix multiplication accumulates in `f32`, mirroring the GPU, where cuBLAS accumulates in `f32`. `std` has no `f32::erf`, so `erf` is computed with the Abramowitz & Stegun 7.1.26 polynomial evaluated in `f64` (max absolute error `1.5e-7`, below `f32` resolution). The backend allocates a fresh tensor per op, parallelises matmuls across output rows with `std::thread::scope` (no external crates), and uses a `k`-outer loop for `matmul` and a per-output dot product for `matmul_t` so both operands stream sequentially.

The MLX backend also computes in `f32` end to end; no `f16` accumulation is used anywhere, deliberately. Two readback details are load-bearing: `download_f32` calls `contiguous` before reading, because an MLX array is frequently a strided view (a transpose, an integer index) and the flat readback is only defined for row-contiguous buffers, and `sync()` is a real device barrier rather than a no-op (see the MLX gotchas under **Backends**).

**No backend's `weight_format()` selects `WeightFormat::F16`**, because for a backend that computes in `f32` a resident `f16` weight is a narrowing that buys nothing. `Tensors::f32` is the only way weights leave the file: it widens `F16` to `f32` unconditionally, so every backend computes from `f32` weights and no backend's parity is affected by the file being `F16`. **That is why the GGUF can store `F16` while this policy still holds**: the narrowing happens in the file, not in the arithmetic, and no tolerance was widened to admit it. Weight loading uploads one tensor at a time and drops the host buffer immediately, so peak host memory stays near the resident size rather than twice it. `Tensors::f32_shaped` returns flat `f32` data together with its *effective* shape — the shape that matches the returned buffer once the layout fix-up has been applied.

## Golden fixtures and tolerance

One JSON document per scenario, written by `golden`, replayed by `tests/golden.rs`:

```json
{
  "name": "demo",
  "backend": "cuda",
  "dtype": "fp32",
  "request": { "...the original System One request..." },
  "forwards": [
    {
      "n": 3, "l": 58, "k": 3,
      "ids": [], "att": [], "mpos": [], "mmask": [], "qtype": [],
      "rows": [ { "id": "department", "markers": 3, "qtype": 0 } ],
      "logits": [], "act": []
    }
  ],
  "answers": { "department": { "..." : "..." } }
}
```

A scenario may produce several forwards (the `many-questions` fixture produces two, because its 12 questions do not fit one `--max-tokens 300` group), so `forwards` is an array. `rows` maps batch rows back to question ids, which is what lets the test re-score from replayed logits and compare the stored `answers`. Floats are stored as JSON numbers and round-trip `f32` bit-exactly. Fixtures are committed, so `tests/golden.rs` runs on any machine that has the weights; the scenario set lives in `tests/scenarios.json`.

Raw `logits` and `act` are **unbounded** — the act head's final projection reaches ~5e3 — so an absolute tolerance on them is meaningless. Parity there is judged as `|got - want| <= ATOL + RTOL * |want|` with `ATOL = 1e-3` and `RTOL = 1e-4`. The *contract* is enforced separately and absolutely: calibrated probabilities, confidences, scores and `noul` values all live in `[0, 1]` and must agree to `1e-3` (`PTOL`).

Measured agreement on the committed fixtures (6 forwards, 19 scored rows, 64 raw logits), each backend replaying the CUDA-recorded output. The `cpu` column is Linux / RTX 3060 host, 20 threads, reading the `F16` GGUF; the `mlx` column was measured on Apple M4.

| backend | quantity | worst observed | limit |
|---|---|---|---|
| `cuda` | raw logits | `0.000e0` absolute | `1e-3` + `1e-4`·\|want\| |
| `cuda` | raw act | `0.000e0` absolute | same |
| `cuda` | calibrated answers | `5.6e-17` | `1e-3` |
| `cpu` | raw logits | `2.2e-5` absolute / `1.9e-5` relative | `1e-3` + `1e-4`·\|want\| |
| `cpu` | raw act | `1.0e-2` absolute / `2.1e-6` relative | same |
| `cpu` | calibrated answers | `1.0e-4` | `1e-3` |
| `mlx` | raw logits | `2.5e-5` absolute / `4.5e-5` relative | `1e-3` + `1e-4`·\|want\| |
| `mlx` | raw act | `2.4e-3` absolute / `5.4e-7` relative | same |
| `mlx` | calibrated answers | `1.0e-4` | `1e-3` |

`cuda` replays the fixtures it recorded and does so bit-exactly (`|dlogit| = 0`, `|dact| = 0`): that is the check that the GGUF loader, the layout fix-up and the fused forward agree with the export the fixtures were recorded from.

**Do not read a calibrated figure below `1e-4` as evidence of anything.** `confidence` is a normalised entropy and the API rounds every answer to 4 decimals, so a backend that is only *nearly* bit-identical can land either side of a rounding boundary and show up as a full `1e-4` step: `single-choice / dept` makes `cuda` evaluate `confidence` to `0.694150941` and `cpu` to `0.694149852`, straddling `0.69415` by less than `1e-6`. One rounding step is the honest bound for such a backend, and it is still an order of magnitude inside `PTOL`. `cuda` recorded the fixtures, so its `confidence` values never cross a boundary and its `5.6e-17` is a different thing entirely.

No tolerance was widened to admit MLX: it is held to the same `ATOL`/`RTOL`/`PTOL` constants the CPU backend is, and passes with roughly two orders of magnitude of headroom on the calibrated contract.

**A hand-written `f16` GEMM behind `--gemm f16` was deleted on 2026-09-23; do not bring it back.** It rounded *activations* to `f16` at every one of the 120 GEMMs in the 28 encoder layers, which accumulated to 4.65x the raw-logit tolerance (worst `5.8e-3` against a `1.3e-3` limit, 26 of 64 assertions over; calibrated answers `1.4e-3` against `PTOL = 1e-3`), and it was *slower* than cuBLAS on the hardware it was written for (`p50` 53.1 ms against 30.3 ms). The only axis it won was residency and startup (load 3.57 s → 0.67 s), which does not justify an unverified kernel that misses the project's own accuracy bar. `--gemm` is rejected at the CLI with that reasoning rather than silently accepted, and the `PKind`/`Packed`/`Tensors::packed` machinery that fed it went with it.

## Batching

Questions are packed into as few forwards as the `--max-tokens` budget allows. Items are sorted by sequence length and accumulated greedily: a new item joins the current group unless `max(len_so_far, new_len) * (group_size + 1)` would exceed the budget. Every sequence in a group is padded to the group's longest, with `att = 0` marking padding so the mask can drop those keys.

## Request shape (System One)

`systemone` owns everything between the JSON and the graph: it parses the request, renders each question's options as text, builds the sequences, and shapes the answers. Backends only ever see marker scores and act logits.

```json
{
  "state": "Help! My payouts have failed for 3 days.",
  "questions": {
    "dept":   { "type": "choice", "instructions": "Which team should handle this?",
                "criteria": { "billing": "Payments", "technical": "Bugs" } },
    "score":  { "type": "score",  "instructions": "How severe is this?",
                "criteria": ["cosmetic", "degraded", "outage"] },
    "urgent": { "type": "noul",   "instructions": "Does this convey urgency?",
                "criteria": { "true": "yes, it is urgent", "false": "no" } }
  }
}
```

* `choice` criteria is a map of option to description — or a plain array of options, which is normalised to that map with `null` descriptions. One marker per option.
* `score` criteria is an ordered array of 2..10 level descriptions, rendered as `level i: <text>`.
* `noul` criteria is optional; `true` / `false` carry defaults when absent. It is a two-marker question whose answer is the probability of "true".
* `state` is optional, and both the state and the option text have `[MASK]` replaced by a space before tokenization (`serialize_state`).

Each question becomes one sequence: `[CLS] "<type> question: <instructions>" [SEP]`, then one marker per option (a mask token followed by that option's first ≤ 48 tokens), then `[SEP]`, the state, and a final `[SEP]`. The question head plus all option markers share a `HEAD_MAX_LEN = 192` budget (the options are re-truncated per option if they overflow it) and the whole sequence is truncated to `MAX_LEN = 512`. The marker positions and the `qtype` (`choice` 0, `score` 1, `noul` 2) are handed to `graph::forward` as index slices (see **Op contract**), and calibration is applied afterwards, in `systemone`, never in a backend.

## Backends

### CUDA

The CUDA backend deliberately does not implement `ops::Backend`. It stays a hand-written fused forward pass (NVRTC-compiled kernels launched directly, cuBLAS for the dense matmuls) and is the latency reference; routing it through per-op calls would cost more than it buys. `tests/golden.rs` is what keeps `src/graph.rs` and `src/backend/cuda/mod.rs` in agreement, which is why the fixtures are the contract rather than an implementation detail. Requires CUDA 13.x (with cuBLAS and NVRTC), Rust 1.85+, and a GPU with ≥ 2 GB VRAM (measured peak: 1.8 GB).

### MLX

`src/backend/mlx/mod.rs` implements all 16 `ops::Backend` ops and `engine::Engine`, so the whole model comes from `graph::forward`, exactly as for the CPU backend. Parity is measured, not asserted: `mlx` replays all 6 forwards / 19 scored rows of the committed fixtures inside the unchanged `ATOL = 1e-3` / `RTOL = 1e-4` / `PTOL = 1e-3` policy, and the 18 op tests run against both backends from a single macro body.

MLX requires Apple Silicon, macOS, `cmake`, and network access on the first build. `mlx-sys` 0.6 forces `MLX_BUILD_METAL=ON` and passes `-DMLX_METAL_PATH`, so the `mlx` feature cannot build on Linux at all: CMake fails with `NOTFOUND` before any Rust code is compiled. This is a deliberate platform boundary, not a bug. `mlx-sys` does not use a system MLX either: its `build.rs` runs CMake against the vendored `mlx-c` tree, which `FetchContent`s Apple's MLX source and builds it together with `mlx-c` (version 0.32.2), so a `brew install mlx mlx-c` is irrelevant — those libraries are never linked. The Metal library lands in `~/.mlx/lib/<hash>/mlx.metallib` (override with `MLX_RS_METAL_PATH`); if it is missing, the build warns and Metal ops fail at runtime rather than at link time. `safetensors` 0.8 is a third `mlx`-gated direct dependency, used to read and write the quantised weight file, because `mlx-rs` exposes only the `Array` ↔ `TensorView` conversions, not the container.

Details that cost real time:

* **The mask must be broadcast to rank 4.** SDPA requires rank-4 `q`/`k`/`v` and validates the mask against the score shape; a `[n, l, l]` mask broadcasts against `[n, nh, l, l]` only when `n == 1`, which is the batch size every op-level test uses. The committed fixtures, at `n = 3`, fail with `Shapes (3,58,58) and (3,16,58,58) cannot be broadcast`. One `expand_dims(1)` fixes it, giving `[n, 1, l, l]` — the argument for a committed fixture set in one line.
* **The additive float mask is the right choice.** MLX's fallback SDPA does `scores = add(scores, mask)` for a non-boolean mask and `where(mask, scores, -inf)` for a boolean one, so the model's existing `0.0` / `-3.4e38` float mask reproduces the CPU reference exactly, and `-3.4e38` stays finite so the softmax never sees an infinity.
* **`split_at_indices` keeps the split axis.** Splitting `[n, l, 3, nh, hd]` into three at `[1, 2]` yields `[n, l, 1, nh, hd]`, not `[n, l, nh, hd]`; each part needs an explicit `squeeze_axes(&[2])`. A size-1 axis also survives a `swap_axes` silently, so this surfaces as a shape assertion rather than a value error.
* **Avoid `fast.rope`.** The rotate-half pairing is expressed explicitly as `concat([x1*cos - x2*sin, x2*cos + x1*sin])` over a split of the head dimension, fed from the graph's `rope_tables`, so the model's own table is used and no synthetic `1/10000^(2i/d)` schedule is invented.
* **`mlx-rs` 0.32 exposes no synchronise primitive** and does not re-export `mlx-sys`'s bindings, so `mlx-sys = "=0.6.0"` is a feature-gated direct dependency purely to call `mlx_synchronize` on the thread-local stream.
* **GELU is the exact `erf` form** — `0.5 * x * (1 + erf(x / sqrt(2)))` using MLX's `erf`, not an approximation. The CPU backend computes `erf` with a polynomial in `f64`; the two agree far inside tolerance.

Steady state, `bench --warmup 10 --iters 50` on the committed `demo` scenario (3 questions, `l = 58`, `k = 3`, one forward). `cuda` is an RTX 3060 Laptop (Linux, CUDA 13.4) reading the `F16` GGUF; `mlx` and `cpu` are Apple M4 / 16 GB, so the three are directly comparable:

| backend | p50 | p90 | mean | notes |
|---|---|---|---|---|
| `cuda` | 30.3 ms | 31.5 ms | 30.7 ms | 10.2 ms/question, 98 q/s, weights load in 3.57 s |
| `mlx` | 62.1 ms | 62.7 ms | 62.2 ms | 20.7 ms/question, 48 q/s, weights load in 0.30 s |
| `cpu` | 1267.9 ms | 1273.8 ms | 1265.7 ms | same request, `--warmup 3 --iters 10` |

The CUDA row is **load-bound at startup, not compute-bound**: 3.57 s to load against 30.3 ms per forward. Most of that is the layout fix-up and the widening it feeds — the CUDA path widens the whole `F16` file to `F32` and physically transposes the 120 `matmul`-family tensors on the host before uploading, so it moves 1.7 GB to the device. That is a one-time cost per process, and the price of keeping `graph.rs` on its original weight layouts.

### Quantisation

`ops::Backend` carries a weight-precision protocol. A backend declares, per tensor, what it wants via `weight_format(name, dims) -> WeightFormat`, and `graph::load` dispatches on that declaration through `upload_f32` / `upload_f16` / `upload_quant`; `resident(name)` lets a backend say it already holds the packed form so the graph skips the f32 read entirely. Every method has a default that degrades to `upload_f32`, so the CPU backend and the op tests are untouched by the protocol's existence.

MLX affine quantisation is produced by `convert`, which writes a standard **safetensors** file (read back through the `safetensors` crate's `SafeTensors`/`TensorView` and mlx-rs's `Array` conversions). It reads the GGUF, not the checkpoint:

```bash
laya convert --to mlx --bits 4 --group 64     # mlx build; writes models/laya/mlx/weights.safetensors
laya test  --quantized - < req.json           # mlx: use the quantised weights
laya bench --quantized --warmup 10 --iters 50 - < req.json
```

`--quantized` on the MLX backend means "MLX affine-quantised weights"; it resolves to `mlx/weights.safetensors` next to the GGUF, and the dense tensor values still come from that GGUF. If the file is absent the flag falls back to the dense backend rather than quantising at load. `bits`/`group` live in the file's safetensors metadata, so the file is self-describing and `--bits`/`--group` only matter when converting. At runtime the weights feed `quantized_matmul`: `matmul` and `matmul_t` both quantise the GGUF weight as stored along its last axis and then differ only in the `transpose` flag, so no host-side transpose or re-layout is needed. `embedding` and `add_type` gather rows from the packed weight, scales and biases and `dequantize` the gathered slice.

Details that turned out to matter:

* **MLX accepts only group sizes 32, 64 and 128**, and the group must divide the last axis of the weight. This is the single most restrictive constraint: of the 207 weight tensors, 124 are eligible at group 64. `act_head.0.weight` is `[256, 1028]` and 1028 is not divisible by 32, 64 or 128, so the act head's first projection cannot be quantised at all and stays `f32`. `scorer.3.weight` (the scorer output, `[1, 1024]`) is likewise excluded. Together they are 352k of 421.3M parameters.
* **Quantising the token embedding is counter-intuitive but correct**: excluding it (a plausible "embeddings are sensitive" rule) made 4-bit agreement *worse*, 62.5% against 68.75%, and raised resident memory from 251 MB to 437 MB because the table then stays `f32`. It is quantised.
* **No `PKind` re-pack is needed.** `convert` requantises from `f32` with MLX's own `quantize`, so the packed bytes are the reference implementation's by construction and there is no bit-order surface to get wrong.

Measured on Apple M4 16 GB — `demo` scenario for latency and memory, all 5 committed scenarios (19 scored rows, 16 of them `choice`) for agreement against dense weights:

| config | file | peak RSS | load | p50 | p90 | mean | choice agreement | max prob. drift |
|---|---|---|---|---|---|---|---|---|
| fp32 | 1685 MB onnx | 2131 MB | 288 ms | 57.4 ms | 57.5 ms | 57.4 ms | 16/16 = 100% | — |
| 8-bit g64 | 452 MB | 967 MB | 93 ms | 60.0 ms | 60.1 ms | 60.0 ms | 15/16 = 93.75% | 0.0238 |
| 4-bit g64 | 251 MB | 565 MB | 60 ms | 59.9 ms | 60.1 ms | 59.8 ms | 11/16 = 68.75% | 0.6522 |

**Memory and load time are the wins; latency is not.** 4-bit takes 2131 MB → 565 MB (3.8×) and 288 ms → 60 ms; 8-bit takes 967 MB (2.2×). Per-forward latency gets slightly *worse* (57.4 ms → 59.9 ms), because at 3 questions and 58 tokens the forward is dominated by the 28 sequential encoder layers and by dequantisation overhead inside `quantized_matmul`, not by weight bandwidth. Quantisation here buys footprint and startup, not speed.

**Neither configuration meets the `choice` agreement ≥ 99% bar, and no lossy configuration can on this corpus.** With 16 choice rows, 99% means 16/16, and 8 of those rows are near-ties *in the fp32 reference itself* (top-2 margins from `0.0008` to `0.02`): `many-questions` asks the same synthetic question twelve times and the model's answer is a coin flip on most of them (fp32 reads `alpha 4 = 0.3555` against `beta 4 = 0.3547`). The 8-bit run perturbs probabilities by at most `0.0238`, and its single disagreement is a `0.0015`-margin row, so that flip is a near-tie crossing zero rather than damage to a real decision. Configurations tried on the same 16 rows: 4-bit g32 56.25%, 4-bit g64 68.75%, 8-bit g32 93.75%, 8-bit g64 93.75%. **A quantisation acceptance test needs a corpus with real decision margins; on this one the metric measures tie-breaking.**

No fixture, tolerance, scenario or assertion was changed to produce these numbers; the quantised paths are measured against the fp32 backend rather than against the CUDA fixtures, which record fp32.

## Layout

| Path | Role |
|---|---|
| `src/lib.rs` | Library root, feature guard, `model_spec` / `tokenizer_json` / `floats` helpers |
| `src/arch.rs` | Backend-free model description: constants, weight names, the transposed-tensor set |
| `src/gguf.rs` | GGUF v3 container: metadata value types, tensor table, alignment — read and write, no dependencies |
| `src/convert.rs` | `convert --to gguf`: checkpoint + configs + tokenizer into one GGUF, including the RoPE tables |
| `src/engine.rs` | Coarse seam: `Req`/`Out`, `ModelSpec`, `EngineInfo`, the object-safe `Engine` trait |
| `src/ops.rs` | Fine seam: the `Backend` op trait, the `WeightFormat` weight-storage protocol, and `TapSink` |
| `src/graph.rs` | The model, written once, generic over `Backend` |
| `src/backend/mod.rs` | Backend registry and feature gating |
| `src/backend/cuda/mod.rs` | CUDA backend: fused forward pass |
| `src/backend/cuda/kernels.rs` | CUDA C kernel source, compiled at runtime by NVRTC |
| `src/backend/cpu/mod.rs` | CPU reference backend implementing the op trait |
| `src/backend/mlx/mod.rs` | MLX backend implementing the op trait (Apple Silicon) |
| `src/weights.rs` | `Tensors` (GGUF-backed, with the layout fix-up) and the safetensors reader `convert` uses |
| `src/systemone.rs` | Request parsing, sequence construction, calibration, answer shaping |
| `src/tokenizer.rs` | BPE tokenizer (pure CPU) |
| `src/main.rs` | CLI front-end: `clap`-derived subcommands whose flags are scoped and typed per command, including the `serve` HTTP loop and its inference thread; everything model-related lives in the library |
| `tests/ops.rs` | Op unit tests (every compiled backend, no weights) |
| `tests/gguf.rs` | GGUF container tests: metadata types, tensor table, alignment (no weights) |
| `tests/golden.rs` | Golden replay: regression on cuda, parity on cpu / mlx |
| `tests/fixtures/`, `tests/scenarios.json` | The golden contract |

`engine::Engine` is the coarse, object-safe seam. It returns host memory (`Out { logits: Vec<f32>, act: Vec<f32> }`) and boxes cleanly, so `--backend` dispatch is a `match` and a `Box<dyn Engine>`. Returning host memory is deliberate: it forces a lazy backend to materialise before returning, so neither timing nor scoring can accidentally measure graph construction. `EngineInfo` exists so `--verbose` can report what actually ran and so backend-specific flags can be rejected *before* an expensive model load.

`ops::Backend` is the fine-grained seam, generic over an associated tensor type, so a backend keeps native handles with no dynamic dispatch in the hot path. Index inputs (`ids`, `att`, `mpos`, `mmask`, `qtype`) are passed as host slices rather than tensors: they are tiny, they already live in the request, and it keeps an integer tensor type out of the abstraction.

Weight-source resolution is trivial now that there is one format: `model_path` resolves `--model`, else `$LAYA_MODEL`, else `models/laya`, `gguf_path` passes an existing file through and otherwise appends `laya-f16.gguf` to the directory, and the only variant is the MLX affine file next to the GGUF, selected by `--quantized`. `EngineInfo.dtype` reports what the GGUF actually declares rather than what a flag asked for, so `--verbose` cannot lie about it.

Adding a backend means adding `src/backend/<name>/`, its cargo feature, one entry in `backend::ALL`, `backend::compiled`, `backend::open`, and one arm in the `open_<name>` functions.
