# QORA

Pure Rust text-only CPU inference runtime for Qwen3.5-4B.

No Python, no CUDA, no external ML frameworks. A single executable
and model weights provide portable AI inference on Linux x86_64
processors with AVX2 support.

> [!NOTE]
> *Rust* 🦀 inside!


## What It Does

QORA (Quantum Orchestrated Runtime Architecture) is a 4-billion-parameter
language model based on Qwen3.5-4B.
This project provides a pure-Rust, text-only CPU inference runtime.

It supports:

- **Text generation** — answer questions, write code, and explain technical concepts
- **Thinking mode** — extended reasoning with a configurable token budget
- **Greedy decoding** — deterministic token selection
- **Streaming generation** — display generated text as inference progresses


## Architecture

QORA uses a hybrid architecture combining two attention mechanisms:

| Component | Details |
|-----------|---------|
| **Parameters** | 4B total |
| **Hidden dim** | 2560 |
| **Layers** | 32 (24 DeltaNet + 8 Full Attention) |
| **Layer pattern** | 3x DeltaNet + 1x Full Attention, repeated 8 times |
| **Vocabulary** | 248,320 tokens |

### DeltaNet Layers (24 of 32)
- Gated linear attention with delta rule state updates
- 16 QK heads + 32 V heads, head_dim=128
- Causal Conv1d (kernel=4) + SiLU activation
- Constant-size recurrent state per DeltaNet layer, without
  a conventional per-token KV cache for those layers

### Full Attention Layers (8 of 32)
- Grouped Query Attention (16Q / 4KV heads), head_dim=256
- QK-norm + partial RoPE (64/256 dims rotated), theta=10M
- Output gating (sigmoid gate on attention output)
- Standard KV cache


## Smart System Awareness

QORA detects your system at startup and automatically adjusts generation limits:

```
QORA - Pure Rust Text-Only CPU Inference Engine
System: 64037 MB RAM (58860 MB free), 12 threads
```

| Available RAM | Think Budget | Max Tokens | Behavior |
|---------------|-------------|------------|----------|
| < 4 GB | 128 (cap 256) | 256 (cap 512) | Minimal generation, warning displayed |
| 4-8 GB | 256 (cap 1024) | 512 (cap 1024) | Constrained, warning displayed |
| 8-12 GB | 1024 (cap 2048) | 1024 (cap 2048) | Normal operation |
| >= 12 GB | 2048 (cap 8192) | 2048 (cap 8192) | Full capability |

Hard caps apply even to explicit user values on **Linux** /proc/meminfo.


### AVX2 SIMD Acceleration

QORA includes AVX2-optimized CPU kernels for Q4 and F16 operations,
with scalar reference implementations.

The AVX2 implementations are tested against their scalar equivalents.
Runtime dispatch selects an appropriate implementation on supported
x86_64 processors.

Multi-threaded execution uses Rayon.

Current performance: approximately 7 tok/s prefill and 2.4 tok/s decode.


## Quick Start

1. Download from [Releases](https://github.com/qora-protocol/QORA-LLM-4B/releases) page:
   - `model.qor4b.part_aa` + `model.qor4b.part_ab` (reassemble below)
   - `tokenizer.json`

2. Reassemble the model file:

**Linux:**
```bash
cat model.qor4b.part_aa model.qor4b.part_ab > model.qor4b
```

3. Build from source:

```bash
cargo build --release
```

4. Usage:

```bash
echo "2 + 3?" | ./target/release/qora --show-think --verbose
```

```bash
RAYON_NUM_THREADS=10 ./target/release/qora --no-think --greedy \
  < prompt.md > generated.txt 
```

```bash
RAYON_NUM_THREADS=10 ./target/release/qora --no-think --greedy \
  --verbose < prompt.md > generated.txt 2> diagnostics.log
```

```bash
RAYON_NUM_THREADS=10 ./target/release/qora --prompt "2 + 3?" \
  --no-think --greedy
```

```bash
RAYON_NUM_THREADS=10 ./target/release/qora \
  --prompt "2 + 3?" --no-think --greedy
```

```bash
RAYON_NUM_THREADS=10 ./target/release/qora \
  --show-think --verbose \
  < <(echo "2 + 3?") \
  > generated.txt \
  2> diagnostics.log
```

```bash
cat <<'EOF' | RAYON_NUM_THREADS=10 ./target/release/qora --no-think --greedy
You are a helpful technical assistant.

Explain:
1. CPU cache
2. Cache line
3. Cache coherence

Use simple technical language.
EOF
```

NOTE: System diagnostics report available system resources and logical CPUs.
Rayon worker count can be configured independently using `RAYON_NUM_THREADS`.


### CLI Flags

```bash
➜  qora-text-avx2-linux git:(main) ✗ ./target/release/qora --help   
Usage: qora [--prompt <prompt>] [--max-tokens <max-tokens>]
  [--think-budget <think-budget>] [--no-think] [--greedy] [--show-think]
  [--verbose] [-V]

QORA text-only CPU inference engine

Options:
  --prompt          prompt text (takes precedence over stdin when both are
                    supplied)
  --max-tokens      maximum number of generated tokens
  --think-budget    thinking token budget
  --no-think        disable thinking
  --greedy          use greedy decoding (temp. = 0)
  --show-think      display thinking tokens
  --verbose         print diagnostic and progress information to stderr
  -V, --version     print version information
  --help, help      display usage information

➜  qora-text-avx2-linux git:(main) ✗
```


### Sampling Defaults

| Parameter | Think mode | No-think mode |
|-----------|-----------|---------------|
| temperature | 1.0 | 0.7 |
| top_k | 20 | 20 |
| top_p | 0.95 | 0.95 |
| presence_penalty | 1.5 | 1.5 |


## File Structure

```
src/
  main.rs           — CLI entry point, argument parsing
  config.rs         — Model architecture configuration
  gemv.rs           — Fast GEMV/GEMM for QORA hybrid architecture (Gated DeltaNet + Full Attention)
  simd.rs           — Runtime-dispatched AVX2 CPU kernels for QORA text inference
  generate.rs       — Text generation loop (text, image, video modes)
  tokenizer.rs      — Tokenizer wrapper and chat templates
  save.rs           — Binary model format (.qor4b) save/load
  convert.rs        — One-time safetensors → .qor4b converter
  system.rs         — System awareness (RAM detection, smart limits)
  lib.rs            — Module exports
```


## Model Binary Format

Custom `.qor4b` binary format for fast loading:

```
Header:  "QOR4" magic + version(u32) + format(u8: 0=F16, 1=Q4)
Config:  Architecture params (vocab, hidden, layers, heads, etc.)
Layers:  32 layers, each with type byte + layer-specific weights
Global:  Embedding + final norm + precomputed RoPE tables
Vision:  Conv3d patch embed + pos_embed + 24 ViT blocks + merger MLP
```

Q4 model loading takes approximately 1 second on the reference system (Intel Core i7-1255U,
12 logical threads, AVX2, 64 GB RAM, passive cooling).


## License

This project is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
The base model [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) is released under Apache 2.0.
