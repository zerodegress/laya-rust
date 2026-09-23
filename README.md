# laya-rust

Rust inference engine for [Laya](https://huggingface.co/convaiinnovations/laya), without graph optimisation. It takes the [TypeSafe System One](https://docs.typesafe.ai/api) request shape — a `state` plus typed `questions` — and returns typed `answers`.

The crate is both a library and a CLI. Inference backends are pluggable, and exactly one is compiled per build:

| Backend | Notes |
|---|---|
| `cuda` | fused forward pass, cuBLAS matmuls, NVRTC kernels; the latency reference (default feature) |
| `cpu` | reference implementation of the op trait; runs on any machine, roughly two orders of magnitude slower |
| `mlx` | Apple Silicon, fp32, optional affine 4/8-bit weights |

Built with no backend feature at all, the build fails with an error naming the available ones.

## Build

```bash
cargo build --release                                        # CUDA (default)
cargo build --release --no-default-features --features cpu    # CPU reference backend
cargo build --release --no-default-features --features mlx    # Apple Silicon only
```

The package is `laya-rust`; the binary it builds is `laya`, which is the name the commands below use.

## Model

The engine reads one file, named at run time: `--model PATH` / `-m`, else `$LAYA_MODEL`, else `models/laya`. A directory means the `laya-f16.gguf` inside it; a `.gguf` path is used as given. Write it once from the upstream checkpoint with

```bash
laya convert --to gguf --model models/laya
```

after dropping the checkpoint's `model.safetensors`, `encoder/config.json`, `rl_agent_config.json` and `tokenizer/tokenizer.json` into the model directory (default `models/laya`). The GGUF carries the weights, the architecture hyper-parameters, the tokenizer and the calibration temperatures in one self-describing file, so nothing else is needed afterwards. `--f32` writes `laya-f32.gguf` instead, and the engine never depends on either name: point `--model` at the file you built. Weights are not part of this repository.

## Usage

```bash
laya test -v '{"state":"Help! My payouts have failed for 3 days.","questions":{"dept":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments","technical":"Bugs"}},"urgent":{"type":"noul","instructions":"Does this convey urgency?"}}}'
laya test  -  < req.json                  # request from stdin
laya test -m models/laya/laya-f16.gguf - < req.json   # or a specific .gguf (env: $LAYA_MODEL)
laya test --backend cpu - < req.json      # pick a backend
laya bench --warmup 10 --iters 50 - < req.json
laya tokenize  -  < req.json              # tokenization only, no backend, no weights
laya weights                              # dump the GGUF tensor table
laya convert --to mlx --bits 4 --group 64 # MLX affine weights (mlx builds)
```

`test` runs one request and prints the System One response, `bench` reports the steady-state latency distribution, `tokenize` dumps the tokenized sequences, `weights` dumps the tensor table read back from the GGUF, and `convert` writes a GGUF from the safetensors checkpoint or an MLX affine weight file from the GGUF. `-h` / `--help`, `help <command>` and `-V` behave as usual; flags are scoped to the subcommand that uses them.

Common flags: `--model PATH` / `-m` (a model directory, or a `.gguf` file; default `$LAYA_MODEL`, else `models/laya`), `--backend auto|cuda|mlx|cpu`, `--max-tokens N` (padded-token budget per forward, default 16384), `--verbose` / `-v`. `-` (or no argument) reads the request from stdin, and empty stdin uses a builtin demo.

Environment: `CUDA_PATH` / `CUDA_HOME` override the NVRTC include path, `LAYA_CPU_THREADS` sets the CPU backend's thread count, `LAYA_MODEL` supplies the default for `--model`.

## Test

```bash
cargo test --release                                          # cuda  (regression vs fixtures)
cargo test --release --no-default-features --features cpu      # cpu   (parity vs fixtures)
cargo test --release --no-default-features --features mlx      # mlx   (parity vs fixtures)
```

The op and GGUF tests need no weights and run anywhere; the golden replay skips itself unless a model is present — a directory holding the GGUF (`LAYA_MODEL_DIR`, or `models/laya`), or a `.gguf` file.

## Requirements

**CUDA backend** — CUDA 13.x (with cuBLAS and NVRTC), Rust 1.85+, a GPU with ≥ 2 GB VRAM (measured peak: 1.8 GB).

**CPU backend** — nothing beyond Rust; expect ~1.7 GB resident, since it widens the GGUF to `f32` at load.

**MLX backend** — Apple Silicon, macOS, `cmake` and network access on the first build; the feature cannot build on Linux at all.

## Design

Two seams, on purpose. `engine::Engine` is coarse and object-safe (`Req` in, `Out { logits, act }` out), which is what `--backend` dispatch and timing go through. `ops::Backend` is fine-grained and generic over a backend's native tensor type, so a new backend inherits the whole architecture by implementing the op trait and gets no dynamic dispatch in the hot path. The CUDA backend deliberately skips the op trait and stays a hand-written fused forward pass; the golden fixtures are what keep it in agreement with the shared graph.

`AGENTS.md` is the developer-facing detail: architecture constants, the GGUF metadata schema, the forward-pass conventions a backend must reproduce exactly, the op contract, the CLI and request-shape reference, the layout, and the measured tolerance, latency and quantisation numbers.
