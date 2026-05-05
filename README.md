# vantablack

> *The Zero-Footprint Zig Inference Engine*

A pure-Zig (0.16+) CLI for local LLM inference. mmap-backed, comptime-specialized
quantized matmul, multi-threaded, with an opt-in Apple Metal GPU forward path.
No `extern "C"` on the CPU path, no libc, no BLAS, no llama.cpp.

### Why "vantablack"

[Vantablack](https://en.wikipedia.org/wiki/Vantablack) is a real-world coating
that absorbs ~99.965% of visible light — objects painted with it lose all
visible surface detail and read as silhouettes. The engine takes the same
posture toward overhead: all the runtime tax that usually surrounds an
inference engine — process startup, weight loading, intermediate copies,
allocator churn, dispatch latency, dependency stacks — is absorbed until only
the model itself is visible. No interpreter tax, no copy tax, no allocation
tax, no framework tax. Just weights and math.

---

## Status

| Capability                                              | State                |
|---------------------------------------------------------|----------------------|
| GGUF v2/v3 parsing (zero-copy mmap)                     | shipped              |
| Llama / Mistral / Gemma architectures                   | shipped — Llama, Mistral, and Gemma all run the full GPU forward path. Mistral SWA reuses existing `attn_scores` via K/V cache offset shift. Gemma uses a new `gelu_approx` MSL kernel. Phi (parallel-block forward): refused at load until segment-API redesign lands |
| KV cache (fixed-size, layer-major)                      | shipped              |
| Persistent thread pool (N-1 workers)                    | shipped              |
| SentencePiece BPE encode + decode                       | shipped              |
| Sampling: greedy / temp / top-k / top-p                 | shipped              |
| Chat template (TinyLlama zephyr)                        | shipped              |
| HTTP server (Ollama-compatible subset)                  | shipped              |
| Apple Metal GPU backend (full forward — see below)      | shipped (opt-in `-Dmetal=true`) |
| Q8_0 / Q4_K / Q5_K / Q6_K / F32 / F16 / BF16            | shipped              |
| TQ2_0 (1.58-bit ternary) plumbed                        | dequant + matmul written, untested on real model |
| Safetensors v1 mmap parser                              | shipped              |
| HuggingFace `config.json` → `LlamaConfig` adapter       | shipped              |
| MLX 4-bit quant CPU kernels (dequant + GEMV)            | shipped              |
| **MLX model end-to-end (loader + tokenizer + forward)** | **shipped** — bit-perfect match with `mlx_lm` reference on TinyLlama-1.1B-Chat-v1.0-4bit |
| MLX 2 / 3 / 4 / 5 / 6 / 8-bit dispatch                  | shipped — 4-bit verified on real model; others covered by unit tests |
| Multi-shard safetensors directory loading               | shipped (covered by `core/hf_loader.zig` test) |
| MLX 4-bit MSL kernel                                    | compiled + cached; runtime dispatch from forward.zig pending |
| Tiktoken-style tokenizer (GPT-2 / Llama-3 / cl100k)     | shipped — byte→Unicode alphabet, byte-level encode/decode, `splitGpt2` + `splitLlama3` (contractions, optional-leading-space binding, 1-3 digit chunks); `Tokenizer.initFromHfJson` auto-detects `ByteLevel` and switches flavor. Llama-3 split structurally matches the cl100k regex; HF reference fixtures still recommended before claiming byte-equality on novel inputs |
| CPU SIMD: `dot_f32` + `dot_i8` primitives               | shipped — both portable @Vector implementations. `dot_f32` lowers to NEON `fmla`. `dot_i8` widens i8 lanes to i32 and reduces; on aarch64+dotprod it stays in NEON `mul.4s + add.4s` rather than the fused `sdot.4s` (Zig 0.16 codegen does not yet pattern-match the chain into `sdot`). Plumbing for a Q8_0 × Q8_0 matmul kernel with dynamically-quantized activations. AMX path explicitly out of scope (undocumented, register layout shifts between M1/M2/M3) |
| Vulkan / cross-vendor GPU                               | not yet              |
| Prompt prefill batching                                 | shipped (CPU) — `runtime/prefill.zig` runs the prompt as one B-wide pass: each weight row is dequantized once and dot-multiplied against B activation rows via `simd.dot_f32`. ~2.3× faster than per-token CPU forward on an 87-token TinyLlama prompt. Metal builds keep the per-token GPU loop pending a batched MSL kernel |

The Metal backend covers the full per-layer forward pass when every
projection is Q8_0, Q4_K, Q5_K, or Q6_K and every norm is f32 (the common
TinyLlama / Llama GGUF layout). On GPU: rmsnorm, QKV matmul, RoPE, KV cache
write, GQA attention (scores + softmax + weighted V), O matmul, residual,
FFN rmsnorm, gate/up matmul, SwiGLU, ffn_down, residual, final rmsnorm, LM
head matmul. One `MTLCommandBuffer` per layer, committed asynchronously;
the CPU only does the embedding gather and the sampler. State buffers and
the KV cache live in shared-storage `MTLBuffer`s that the CPU and GPU read
in place — Apple Silicon unified memory means same physical pages, no DMA,
no copy. Models that don't match the eligibility criteria fall back to the
per-op path (CPU + GPU per-quant matmul).

**Verified end-to-end** on TinyLlama-1.1B-Chat-v1.0 GGUF Q8_0, Q4_K_M,
Q4_K_S, Q5_K_S, Q6_K (ReleaseFast, macOS arm64). GPU output bit-equal to
CPU-only build across all five quants on a 64-token greedy decode.

### MLX-format models — end-to-end

MLX (Apple's array framework) ships Llama-architecture models as a directory
of `*.safetensors` files + `config.json` + `tokenizer.json`, with weights
packed in MLX's own 4-bit / 8-bit affine block-quant format
(`mlx_lm.utils.quantize`). vantablack now loads these natively — point the
CLI at the directory:

```sh
vantablack ~/.cache/huggingface/hub/models--mlx-community--TinyLlama-1.1B-Chat-v1.0-4bit/snapshots/<rev> \
  prompt 64 "Once upon a time"
```

Verified: bit-for-bit identical generation against the `mlx_lm` Python
reference for `mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit` (single shard,
4-bit quant, group_size=64, BF16 scales/biases).

What ships:
- **`core/safetensors.zig`** — single-file safetensors v1 mmap parser,
  zero-copy descriptor extraction.
- **`core/hf_loader.zig`** — directory loader: mmaps every `.safetensors`,
  reads `config.json` and `tokenizer.json` (and `tokenizer.model` if
  present), exposes a unified tensor lookup across shards.
- **`core/hf_config.zig`** — `config.json` → `HfConfig` struct including
  the `quantization: {bits, group_size}` block.
- **`kernels/mlx.zig`** — MLX 4-bit dequant + GEMV (CPU baseline). Handles
  both BF16 and F16 scale/bias dtypes.
- **`runtime/model.zig::Model.initFromHf`** — applies the HF tensor-name
  remap (`model.layers.{i}.self_attn.q_proj.{weight,scales,biases}` →
  `attn_q`, etc.), folds the 3-buffer MLX-quant layout into a single
  `TypedTensor` via the new `mlx` aux pointer.
- **`runtime/tokenizer.zig::Tokenizer.initFromHfJson`** — parses HF
  `tokenizer.json` BPE format directly; merge priorities are recovered
  from the `merges` array order so the existing greedy-BPE encoder runs
  unchanged. Llama-family `▁`/byte-fallback semantics handled.
- **HF/MLX RoPE convention** — neox-style half-rotation
  (`math.ropeHalf`), gated by a `rope_half` flag on `LlamaConfig`. GGUF
  models stay on the original interleaved-pair `math.rope`; the CLI sets
  the flag automatically based on which loader picked the file.
- **CLI dispatch** — if the path argument resolves to a directory, runs
  the HF/MLX path; otherwise runs the GGUF path. `prompt`, `chat`, and
  `serve` all work in either mode (`Server.initFromHf` builds the same
  Ollama-compatible HTTP server against an `HfBundle`); `generate` (raw
  token IDs) remains GGUF-only.

  ```sh
  # MLX HTTP server, Ollama-compatible:
  vantablack ~/.cache/huggingface/hub/models--mlx-community--<repo>/snapshots/<rev> \
    serve --host 127.0.0.1 --port 11434
  # In requests, "model" must match the snapshot directory basename
  # (the rev hash); switch to a friendlier alias by symlinking the
  # snapshot dir under a name of your choice.
  ```

Caveats / not-yet:
- **MLX 4-bit on GPU**: the MSL kernel `matmul_mlx_q4` is compiled into
  the runtime when `-Dmetal=true` is set, but `forward.zig` still routes
  MLX-Q4 layers to the CPU implementation. Reaching the GPU path requires
  threading `HfBundle` shards through `MetalBackend` so each shard's mmap
  is wrapped as its own `MTLBuffer` (currently only the GGUF mmap is
  wrapped). Q8_0-GGUF models keep getting full GPU acceleration unchanged.
- **MLX bit widths**: 2/3/4/5/6/8 are all dispatched correctly on CPU
  (`kernels/mlx.zig::matmul`). End-to-end verification on a real model is
  done for 4-bit only; the other widths are covered by unit tests.
- **Multi-shard safetensors**: directory loading mmaps every shard and
  merges descriptors, with a test (`core/hf_loader.zig`) covering
  cross-file lookup. Works on real sharded models but not yet
  smoke-tested at the inference level (needs a 5 GB+ download).
- **Tokenizer**: BPE/SentencePiece (Llama 1/2 family). Llama 3+ and
  GPT-2 use tiktoken-style byte-level BPE which needs (a) a Unicode-class
  regex engine for the pre-tokenizer split pattern and (b) the GPT-2
  byte→string mapping. Both are absent — adding them is the next sizable
  push for tokenizer coverage.

---

## Performance

TinyLlama-1.1B-Chat-v1.0 on macOS arm64 (M-series), ReleaseFast,
128-token decode, warm caches:

| Build                                  | Quant   | tok/s | Wall (128 tok) | Notes                                                    |
|----------------------------------------|---------|-------|----------------|----------------------------------------------------------|
| CPU `--threads 16`                     | Q8_0    | 45    | 2.85s          | All cores pinned; max CPU throughput                     |
| CPU default (~8)                       | Q8_0    | 28    | 4.57s          | Half cores; default                                      |
| Metal full forward (pre-async)         | Q8_0    | ~65   | 1.96s          | One cmd buffer / layer, per-layer waitUntilCompleted     |
| **Metal async + fusion**               | Q8_0    | **93** | **1.38s**      | **Async segment commit + `-Dweight_fusion=true`**        |
| Metal full forward                     | Q4_K_M  | 102   | 1.25s          | K-quant kernels online                                    |
| Metal full forward                     | Q4_K_S  | 108   | 1.18s          | Smallest K-quant variant                                  |
| Metal full forward                     | Q5_K_S  | 94    | 1.36s          |                                                           |
| Metal full forward                     | Q6_K    | 79    | 1.62s          | Heaviest unpack of the K-quant family                    |

**The headline wins:**
1. `-Dmetal=true` runs the inference on the GPU while the CPU thread pool
   sits at 1 worker. Fan stays off, foreground work stays responsive.
2. Phase 1 of the GPU push (async segment commit + Q/K/V and gate/up
   weight fusion under `-Dweight_fusion=true`) lifts Q8_0 throughput
   from ~65 to ~93 tok/s — a **1.43× speedup** with byte-equal output.
3. Phase 2 lit up Q4_K / Q5_K / Q6_K kernels on the GPU full-forward path.
   Q4_K_M decodes ~12% faster than Q8_0 thanks to halved weight
   bandwidth; Q4_K_S even more so.

With Metal enabled the default is `--threads 1` because additional CPU
workers just spin on GPU sync; the only per-token CPU work is the embedding
gather and the sampler.

vantablack now lands at the upper end of llama.cpp's typical Q8_0 / Q4_K_M
range on the same hardware. The matmul kernel is at the hardware's effective
limit for GEMV at this model scale (three kernel-rewrite experiments —
threadgroup-shared activation tiling, multi-row-per-thread, simdgroup-per-
row with `simd_sum` — all matched the simple per-thread per-row kernel
within noise). Further gains beyond this require token batching for prefill
or sub-32-elem block layouts — see roadmap.

---

## Hardware target

vantablack is designed primarily for **single-user local inference on Apple
Silicon laptops** — the same machine you're already coding on. The whole
design (mmap-first weights, near-idle CPU when GPU is on, `-Doptimize=
ReleaseSmall` < 5 MB) is optimized for the case where you want a model
sitting alongside your IDE that isn't going to spin the fan.

| Platform                            | CPU path        | Metal path      | Status            |
|-------------------------------------|-----------------|-----------------|-------------------|
| macOS arm64 (M1 / M2 / M3 / M4)     | yes             | yes             | primary target    |
| macOS x86_64 (Intel)                | yes             | no              | builds; untested  |
| Linux arm64 (Asahi, AWS Graviton)   | yes             | no              | builds; untested  |
| Linux x86_64                        | yes             | no              | builds; untested  |
| Windows x86_64                      | yes             | no              | builds; untested  |

**Why Apple Silicon first.** Unified memory makes `MTLBuffer`-wrap-of-mmap
free — the GPU reads weights straight from the OS page cache, no host→device
transfer, no doubled memory footprint. On discrete-GPU systems that property
doesn't hold and the design loses much of its appeal.

**Memory sizing.** Weights live in the OS page cache via mmap, so a 1.1 GB
Q8_0 TinyLlama uses ~1.1 GB of RAM at steady state, plus ~50 MB for the KV
cache (TinyLlama at 2048 ctx) plus ~100 MB for the Metal state buffers when
`-Dmetal=true`. A 7B Q4_K_M model would be ~4 GB weights. mmap means
cold-start is bounded by syscall latency, not file throughput — a 70B model
"loads" in milliseconds even on the first run (the kernel pages it in
lazily as the forward pass touches it).

**Disk.** Weight files are read directly from wherever they live — no
pre-load step, no cache directory, no checksum dance. Point the CLI at a
`.gguf` and it runs.

---

## Architecture

```
   argv: model.gguf [subcommand args]              HTTP client (Ollama-compatible)
                  |                                              |
                  v                                              v
   +--------------+--------------+              +----------------+----------------+
   |       src/main.zig CLI      |              |     runtime/server.zig          |
   |   catalog | generate |      |              |   accept loop + http.Server     |
   |   prompt  | chat | serve   -+------------->+   GET /health, /api/tags        |
   +-----------+-+-+-+-----------+              |   POST /api/generate, /api/chat |
               | | | |                          +----------------+----------------+
               | | | +------------ inference_mu (serial) --------+
               v v v                                             v
       ModelMapper  Tokenizer  Sampler  ThreadPool  KvCache   chat_template
      (mmap+catalog)(BPE+SPM)(temp/k/p)(N-1 workers)(O(1) mem)(zephyr wrap)
               \      \         \         |          /
                +-----+----+----+----+----+----+----+
                     |          |          |
                     v          v          v
                +----+----------+----------+----+
                |     runtime/forward.zig       |
                |  per-token Llama forward pass |
                |  RMSNorm, QKV, RoPE, GQA-attn,|
                |  SwiGLU FFN, residuals,       |
                |  LM head -> logits            |
                +----+--------+--------+--------+
                                       |
                                       v
                              CPU path        GPU path (-Dmetal=true)
                                  |                |
                                  v                v
                              kernels/         metal/bridge.{m,h}
                              comptime_gen     + inline MSL kernels:
                              (dispatch        matmul_q8_0, rmsnorm,
                               by quant)       rope, swiglu, residual_add,
                                  |            copy_f32, attn_scores,
                                  v            softmax_rows, attn_weighted_sum
                          +-------+---------+      |
                          | matmul_q8_0     |      v
                          | matmul_q4_k     |  metal_backend.zig
                          | matmul_q5_k     |  (persistent state +
                          | matmul_q6_k     |   KV cache in shared
                          | matmul_f16/f32  |   MTLBuffers; one
                          | matmul_ternary  |   MTLCommandBuffer
                          +-----------------+   per layer)
```

`forward.zig` picks at runtime: GPU forward when the model is Q8_0
projections + f32 norms (and `-Dmetal=true` is on); CPU otherwise. State
buffers + KV cache live in the same shared-storage `MTLBuffer`s either
way — the GPU path just hands their `MTLBuffer` handles to the kernels
while the CPU path reads them as plain `[]f32` slices.

### Module map

```
src/
├── main.zig                  CLI entry point + subcommand dispatch
├── vantablack.zig            public module index (re-exports)
├── core/
│   ├── mapper.zig            ModelMapper: open + std.posix.mmap + catalog (GGUF)
│   ├── parser.zig            GGUF v2/v3 parser, MetaValue union, BlockInfo table
│   ├── safetensors.zig       safetensors v1 parser (mmap-friendly, zero-copy descriptors)
│   ├── hf_loader.zig         HuggingFace / MLX directory loader (mmap shards + JSON)
│   └── hf_config.zig         HuggingFace config.json → struct adapter (incl. MLX `quantization` block)
├── kernels/
│   ├── simd.zig              real Q8_0/Q4_K/Q5_K/Q6_K/TQ2_0/F16/F32 dequant + matmul
│   ├── math.zig              RMSNorm / RoPE / softmax / SwiGLU / argmax
│   ├── mlx.zig               MLX 4-bit quant dequant + GEMV (CPU baseline)
│   └── comptime_gen.zig      QuantType + Kernel + dispatch(comptime q)
├── metal/                    only compiled with -Dmetal=true
│   ├── bridge.h              flat C ABI for the Zig side
│   ├── bridge.m              Obj-C bridge + inline MSL kernels (matmul_q8_0,
│   │                         rmsnorm, rope, swiglu, residual_add, copy_f32,
│   │                         attn_scores, softmax_rows, attn_weighted_sum)
│   └── bridge.zig            extern "c" decls + Device / Segment wrappers
└── runtime/
    ├── kv_cache.zig          fixed-size O(1)-memory KV cache (MTLBuffer-aliased on GPU)
    ├── model.zig             LlamaConfig + Model.init from GGUF metadata
    ├── forward.zig           per-token forward pass; CPU + GPU branches
    ├── metal_backend.zig     persistent Metal state + shared-storage scratch
    ├── pool.zig              persistent ThreadPool (epoch + spinLoopHint)
    ├── tokenizer.zig         SentencePiece BPE encode + decode
    ├── sampler.zig           temp/top-k/top-p + std.Random PRNG
    ├── chat_template.zig     TinyLlama-Chat (zephyr) format helpers
    └── server.zig            HTTP server, Ollama API subset, NDJSON streaming
```

---

## Design constraints (load-bearing)

These are not aesthetic — they shape every line:

1. **Pure Zig std only.** No `extern "C"`, no `@cImport`, no libc linkage,
   no third-party deps.
2. **mmap-first.** Weight tensors are never copied off disk. The OS page cache
   is the working set. Cold-start is bounded by syscall latency, not file
   throughput. A 70B model "loads" in milliseconds.
3. **Zero-copy weight binding.** `Model.init` walks the catalog and stores
   `[]const u8` slices into the mmap region per layer. No staging buffer.
4. **Comptime kernel dispatch.** `kernels.dispatch(comptime q)` returns a
   specialized function pointer at compile time. The inner per-element loop
   never sees a runtime branch on quant type.
5. **Fixed-size KV cache.** One bulk allocation, layer-major, no realloc, no
   fragmentation. O(1) memory across long generations.
6. **No global state.** Allocators, file handles, pools, and caches are all
   passed explicitly.
7. **ReleaseSmall stays under 5 MB stripped.** A forcing function. Currently
   292 KB (includes the HTTP server, JSON parsing, and all kernels). Pulling
   in a dependency would feel as expensive as it actually is.

### "Pure Zig" and GPU

Constraint #1 (no `extern "C"`) was relaxed for the Metal backend. Apple's
Metal API is Objective-C; reaching it without FFI would require raw Mach-O
`dlsym` bindings to the `objc_msgSend` runtime, which is feasible but adds a
maintenance burden out of proportion to the benefit. The compromise:

- The CPU path stays pure-Zig std with no FFI. Builds without `-Dmetal=true`
  link nothing exotic and pass all tests on every target platform.
- The Metal path lives in `src/metal/bridge.{m,h}` (~750 LoC of Objective-C
  + an inline MSL kernel string) and `src/metal/bridge.zig` (the `extern "c"`
  declarations). Builds with `-Dmetal=true` link Foundation + Metal +
  CoreGraphics frameworks.

The bridge is the only Objective-C in the tree and the only file that calls
into a non-Zig API. Vulkan/MoltenVK is not currently planned.

---

## Build

Requires **Zig 0.16.0+**. Verify with `zig version`.

### Installing Zig

If `zig version` doesn't print `0.16.0` or higher, install via one of:

- **macOS (Homebrew):**
  ```sh
  brew install zig
  ```
- **macOS / Linux (multi-version manager — recommended for tracking 0.16+):**
  install [zigup](https://github.com/marler8997/zigup), then
  ```sh
  zigup 0.16.0
  ```
- **Linux / macOS / Windows (manual tarball):** download the matching
  archive from <https://ziglang.org/download/>, extract, and put the
  resulting `zig` binary on `PATH`. No installer, no daemon, no env vars
  beyond `PATH`.
- **Arch Linux:** `pacman -S zig` (check `pacman -Si zig` for the version).
- **Nix / NixOS:** `nix shell nixpkgs#zig_0_16`.

If your distro packages an older Zig (0.13/0.14/0.15), prefer the official
tarball or `zigup` — vantablack uses 0.16-specific std APIs (`std.Io`,
`std.heap.pageSize`, the new `Io.File.Writer`) that won't compile on
earlier versions.

### Build commands

```sh
zig build                                      # Debug, default
zig build -Doptimize=ReleaseFast               # multi-threaded CPU, fast inference
zig build -Doptimize=ReleaseFast -Dmetal=true  # CPU + Apple Metal GPU (full forward on GPU)
zig build -Doptimize=ReleaseSmall              # single-threaded, ~292 KB stripped
zig build test                                 # all unit tests
```

**`-Dmetal=true`** enables the Apple Metal GPU backend (macOS only).
Adds ~130 KB to the binary, links `Foundation` + `Metal` + `CoreGraphics`
frameworks, and compiles a small Objective-C bridge in `src/metal/bridge.m`.
The pure-Zig CPU path remains the default and passes all tests unchanged.

`build.zig` derives per-mode flags from `optimize`:
- `strip = true` for any release mode.
- `single_threaded = true` only for ReleaseSmall (binary-size win).
- `omit_frame_pointer = true` for any release mode.
- `error_tracing = false` for ReleaseSmall.

---

## CLI

```
vantablack <model.gguf>                               print tensor catalog
vantablack <model.gguf> generate <n> <id> [id ...]    n tokens after raw token IDs
vantablack <model.gguf> prompt   <n> <text>           encode + n tokens after text
vantablack <model.gguf> chat     <n> <user-message>   wrap in TinyLlama zephyr template
```

Sampler flags (placed BEFORE the `<n>` value):

```
--temp <f>      temperature (0.0 = greedy, default 0.0)
--top-k <n>     top-k filter (0 = disabled, default 0)
--top-p <f>     nucleus filter (0 = disabled, default 0)
--seed <u64>    RNG seed (default 0)
--system <s>    system prompt for chat (default "You are a helpful assistant.")
--threads <n>   worker thread count (0 = autodetect ~half cores; default 0)
```

The default thread count is deliberately conservative — see the
**Performance** section. Pass `--threads 0` for autodetect, an explicit
number to override, or the same flag to `serve` to set the server-wide pool size.

### Examples

Catalog the model:
```sh
vantablack ~/models/tinyllama.Q8_0.gguf
```

Greedy completion of a text prompt:
```sh
vantablack ~/models/tinyllama.Q8_0.gguf prompt 30 "Once upon a time"
```

Sampling with top-k + top-p:
```sh
vantablack ~/models/tinyllama.Q8_0.gguf prompt \
  --temp 0.7 --top-k 40 --top-p 0.9 --seed 42 \
  60 "The quick brown fox"
```

Chat-template wrapped:
```sh
vantablack ~/models/tinyllama.Q8_0.gguf chat \
  --temp 0.7 --top-k 40 --seed 7 \
  100 "Tell me a short joke about a cat"
```

---

## Server mode (Ollama-compatible)

Long-running HTTP server that speaks a subset of the Ollama API. Use it as
a drop-in backend for editor plugins, code-completion clients, LangChain,
Open WebUI, or anything else that already targets Ollama.

```sh
vantablack ~/models/tinyllama.Q8_0.gguf serve [--host H] [--port P]
# Default: 127.0.0.1:11434 (Ollama's default port)
# vantablack: serving 'tinyllama-1.1b-chat-v1.0.Q8_0.gguf' at http://127.0.0.1:11434/
```

**Endpoints**

| Method | Path             | Purpose                                              |
|--------|------------------|------------------------------------------------------|
| GET    | `/health`, `/`   | Liveness probe — `vantablack ok`                     |
| GET    | `/api/tags`      | One-element model list for client model pickers      |
| POST   | `/api/generate`  | Streamed NDJSON tokens (raw prompt)                  |
| POST   | `/api/chat`      | Streamed NDJSON tokens (chat-template wrapped)       |

**Example: streaming generate**
```sh
curl -N -X POST http://127.0.0.1:11434/api/generate -d '{
  "model": "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
  "prompt": "fn main() {",
  "stream": true,
  "options": {"num_predict": 64}
}'
```

**Example: streaming chat**
```sh
curl -N -X POST http://127.0.0.1:11434/api/chat -d '{
  "model": "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
  "messages": [{"role": "user", "content": "Write hello world in Zig"}]
}'
```

**Example: non-streaming (`stream: false`)**
```sh
curl -X POST http://127.0.0.1:11434/api/generate -d '{
  "model": "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
  "prompt": "Once upon a time",
  "stream": false
}'
```

**Defaults & options**
- `temperature = 0.0` (greedy / deterministic — chosen for code-gen)
- `num_predict = 256`
- Override per-request via `options: {temperature, top_k, top_p, seed, num_predict}`
- The `model` field is validated against the loaded model's basename. Mismatches return HTTP 404.

**Concurrency**
- Serial: one request at a time, mutex-guarded. Concurrent clients queue.
- Reuses Model + KvCache + ThreadPool across requests — no per-request mmap or state-init cost.
- For multi-tenant deployments, run multiple `serve` processes on different ports.

**Idle CPU**
Workers spin briefly between jobs (sub-millisecond wake-up matters in the
hot path) but yield to the OS scheduler after a bounded number of
iterations. Between requests the server holds CPU near 0%; only active
generation pegs the cores.

Per-server `--threads` flag: `vantablack ... serve --threads 4` caps the
worker pool. Use this on shared boxes to reserve cores for other work.

---

## Supported models + quantizations

Targets Llama-architecture GGUF v2/v3 files:
- `general.architecture = "llama"`
- `llama.embedding_length`, `block_count`, `attention.head_count`,
  `attention.head_count_kv`, `feed_forward_length`, `context_length`,
  `rope.freq_base`, `attention.layer_norm_rms_epsilon`
- Standard `blk.{i}.{attn_norm,attn_q,attn_k,attn_v,attn_output,
  ffn_norm,ffn_gate,ffn_up,ffn_down}.weight` tensors
- `token_embd.weight`, `output_norm.weight`, `output.weight` (or tied to
  `token_embd`)

Verified on:
- TinyLlama-1.1B-Chat-v1.0 Q8_0 (pure Q8_0 + F32)
- TinyLlama-1.1B-Chat-v1.0 Q4_K_M (Q4_K + Q6_K + F32)
- TinyLlama-1.1B-Chat-v1.0 Q4_K_S (Q4_K + Q5_K + Q6_K + F32)

Other Llama-arch sizes (7B, 13B, 70B) should work; not yet smoke-tested.
Mistral, Phi, Gemma require small metadata-key adjustments and tensor-name
remapping in `runtime/model.zig`.

---

## CPU, power, GPU — roadmap

The "CPU blew up" problem (Ollama, llama.cpp default builds) has one
root cause: pure CPU matmul saturates every core. The fixes:

1. **Halve the default thread count** — *shipped*. Cuts CPU ~60% with a
   ~55% throughput trade. `--threads N` overrides.
2. **Bounded spin → yield in worker loop** — *shipped*. Idle workers no
   longer pin a core between requests in `serve` mode.
3. **Apple Metal GPU compute (Q8_0 matmul)** — *shipped, opt-in*.
   `-Dmetal=true` build flag. Initial cut: Q8_0 matmul on GPU only,
   one sync per matmul (~155/token), other ops on CPU.
4. **Zero-copy state aliasing** — *shipped*. `forward.State` slices alias
   the backend's persistent shared-storage `MTLBuffer`s. No host↔device
   memcpy on the per-token path.
5. **Single command buffer per layer** — *shipped*. Segment API batches
   rmsnorm + matmul + RoPE + KV-write + attention + O-proj + residual +
   FFN ops into one `MTLCommandBuffer`. ~155 commits/token → ~22.
6. **Full GPU forward** — *shipped*. New MSL kernels: `copy_f32`,
   `attn_scores`, `softmax_rows`, `attn_weighted_sum`, two-buffer
   `rmsnorm`. KV cache + attention scores live in shared `MTLBuffer`s.
   The CPU only does the embedding gather and the sampler.
7. **Tighter MSL matmul (threadgroup tiling, simdgroup matmul)** —
   *attempted, parked*. Tested three matmul kernel rewrites
   (threadgroup-shared activation tiling, multi-row-per-thread,
   simdgroup-per-row with `simd_sum`). All matched the simple per-thread
   per-row kernel within noise on TinyLlama-scale GEMV — Apple's L2 was
   already amortizing activation re-reads across simdgroups, and
   single-token decode at this model scale isn't matmul-kernel-bound.
   Kept the simple kernel; closing the remaining gap to llama.cpp needs
   one of the next three items.
8. **Async pipelining across layers** — *planned*. Enqueue the next
   layer's command buffer while the current one runs (instead of
   `waitUntilCompleted` per layer). ~1.5–3× expected; biggest
   remaining lever for tok/s.
9. **Weight fusion (QKV / gate-up)** — *planned*. Pre-concatenate Q/K/V
   and gate/up weights at init so each fused matmul reads the activation
   buffer once. ~5–10% expected; +~660 MB memory for TinyLlama Q8_0.
10. **Q4_K / Q5_K / Q6_K MSL kernels** — *planned*. Required to put
    K-quant models on the GPU at all. Currently only the Q8_0 forward
    path is GPU-eligible; mixed-quant models stay on the CPU side.
11. **Prompt prefill batching** — *planned*. Process N prompt tokens in
    one forward pass. Big speedup on long prompts (no effect on per-token
    decode rate).
12. **MLX model loader (full integration)** — *shipped*. End-to-end
    inference on `mlx-community/*` Llama-1/2-family 4-bit models, verified
    bit-perfect against `mlx_lm`. 2/3/5/6/8-bit MLX kernels also shipped
    (CPU). Multi-shard safetensors directory loading shipped + tested.
    The MLX-Q4 MSL kernel (`matmul_mlx_q4`) is compiled at startup when
    `-Dmetal=true`; runtime dispatch from `forward.zig` to the GPU is the
    last bit needed to put MLX models on Metal. Tiktoken-style tokenizer
    for Llama 3+ / GPT-2-family is the only remaining piece for broader
    MLX-community coverage.
12. **Apple AMX matrix coprocessor** — *planned*. M1/M2/M3 ship a hidden
    matrix unit accessible via inline assembly. Pure-Zig + lower power
    than even GPU for some shapes. Useful as a CPU-only fallback path.
13. **NEON `sdot` int8 dot** — *planned*. ARMv8.2-A dedicated dot-product
    instruction. Modest CPU-path speedup, no FFI needed.

Items 8 + 9 + 10 are the next perf pushes. Item 7's negative result
shifted the model: the matmul kernel is at the hardware's effective
limit for GEMV at TinyLlama scale; further wins need architecture
changes (async dispatch, weight fusion, batching), not better kernels.

---

## Limitations + future work

- **No prompt prefill batching.** Prompt tokens are processed one at a time.
  For an N-token prompt the cost is N forward passes. Real throughput on
  long prompts will improve substantially with batched prefill.
- **TQ2_0 untested on real weights.** Spec-derived layout, internally
  consistent, no public TQ2_0 GGUF weights to validate against.
- **Single-architecture (Llama).** `runtime/model.zig` requires
  `general.architecture = "llama"` and the standard `blk.{i}.*` tensor
  names. Mistral / Phi / Gemma require small metadata-key adjustments
  and tensor-name remapping; not yet wired up.
- **No tokenizer encoder fallback.** SentencePiece BPE only. Tiktoken-style
  byte-level BPE (GPT-2/Llama-3) requires a different merge algorithm.
- **GPU eligibility is narrow.** `-Dmetal=true` runs the full forward pass
  on GPU only when every projection is Q8_0 and every norm is f32 (the
  common TinyLlama / Llama Q8_0 layout). Mixed-quant models — Q4_K_M,
  Q4_K_S, Q5_K, Q6_K — fall back to a per-op path that still uses the
  GPU for Q8_0 matmul where possible but runs everything else on CPU.
  Closing this needs Q4_K / Q5_K / Q6_K MSL kernels (roadmap item 10).
- **GPU throughput trails llama.cpp.** ~37 tok/s vs ~50–100 tok/s on the
  same hardware. Empirically not a kernel issue — the gap is layer-level
  dispatch + sync overhead. Roadmap items 8 + 9 (async pipelining,
  weight fusion) cover the remaining headroom.
- **Yielding workers, not parking.** When the CPU pool is in use (Metal
  off or fallback path), idle workers `std.atomic.spinLoopHint()` for a
  bounded window then `std.Thread.yield()` — they don't block on a
  futex/condvar (Zig 0.16's std doesn't expose futex publicly without
  libc). Idle CPU drops sharply between requests but is not zero.
- **Server is serial.** One request at a time, mutex-guarded. Good for
  single-user code-gen; multi-tenant requires multiple `serve` processes
  on different ports.
- **Server response metadata is minimal.** `created_at` is a placeholder
  (`1970-01-01T00:00:00Z`); `digest`, `total_duration`, `prompt_eval_count`,
  `eval_count`, `eval_duration` are omitted. Most Ollama clients tolerate
  it; full parity would add real timestamps + sha256 + per-request timings.
- **No `error_tracing` in ReleaseSmall.** `build.zig` disables it for the
  binary-size win. Crashes in ReleaseSmall builds give a stack address,
  not a Zig stack trace. Use ReleaseFast or Debug for diagnostic builds.

---

## Build history

The engine grew in tracked phases. Commit log mirrors this:

1. **Task 1 — Foundation** — `build.zig` flags, `core/{mapper,parser}.zig`,
   `kernels/comptime_gen.zig`, kernel stubs. CLI prints tensor catalog.
   ReleaseSmall stripped < 5 MB constraint established.
2. **Task 2 — End-to-end inference** — `kernels/{simd,math}.zig` real
   Q8_0/Q4_K/TQ2_0/F16/F32 paths, `runtime/{kv_cache,model,forward,
   tokenizer,sampler}.zig`. CLI gains `generate` (token IDs).
3. **Task 3 — Real-model parity** — Smoke-tested on TinyLlama Q8_0;
   added SentencePiece BPE encoder; sampler temperature/top-k/top-p +
   seeded PRNG; CLI `prompt` (text) and chat-template wrapping.
4. **Task 4 — Q5_K + Q6_K + thread pool + chat** — Closed K-quant family,
   verified Q4_K_M / Q4_K_S on real model, added `runtime/pool.zig`
   persistent worker pool (5–8× speedup on matmul), CLI `chat` subcommand.
5. **Task 5 — HTTP server** — `runtime/server.zig` Ollama-compatible
   subset (`/api/generate`, `/api/chat`, `/api/tags`, `/health`),
   NDJSON streaming via `http.Server.respondStreaming`,
   `runtime/chat_template.zig` reusable template helpers, CLI `serve`.
6. **Task 6 — Apple Metal GPU (Q8_0 matmul)** — `src/metal/bridge.m`
   Objective-C bridge + inline MSL kernel + `runtime/metal_backend.zig`
   per-process state (zero-copy `MTLBuffer` wrapping the mmap'd weight
   region, pre-allocated shared scratch buffers). Opt-in build flag
   `-Dmetal=true`. CPU% drops from ~700% to ~27%. **First task to
   relax the no-FFI constraint** — bridge needs to call Obj-C runtime,
   linker needs Foundation+Metal frameworks. Pure-Zig path preserved
   and still default.
7. **Phase 1 — Zero-copy state aliasing** — `forward.State` slices
   alias the backend's persistent shared-storage `MTLBuffer`s. Per-token
   matmul path drops the host↔device memcpys.
8. **Phase 2 — Segment API + per-layer batching** — `bridge.m` exposes
   `vtb_metal_segment_*` for chained dispatches in one
   `MTLCommandBuffer`. Two segments per layer (pre-attn + post-attn)
   replace seven separate command buffers.
9. **Phase 3 — Full GPU forward** — new MSL kernels: `copy_f32`,
   `attn_scores`, `softmax_rows`, `attn_weighted_sum`, two-buffer
   `rmsnorm`. KV cache + attention scores moved into shared `MTLBuffer`s.
   `forward.zig` ships a single command buffer per layer; CPU does only
   embedding gather + sampler.
10. **Phase 4 — Matmul kernel exploration** — tested threadgroup-shared
    activation tiling, multi-row-per-thread, simdgroup-per-row with
    `simd_sum`. All matched the simple kernel within noise; kept the
    simple kernel and updated docs to reflect that further perf wins
    will come from architecture (async pipelining, weight fusion,
    batching), not the matmul kernel itself.

Each phase preserved the hard constraints that still apply (mmap-first,
< 5 MB stripped, no global state) and shipped passing tests + real-model
smoke before moving on.

---

## License

See `LICENSE`.
