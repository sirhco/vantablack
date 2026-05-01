# vantablack

> *The Zero-Footprint Zig Inference Engine*

A pure-Zig (0.16+) CLI for local LLM inference. mmap-backed, comptime-specialized
quantized matmul, multi-threaded, no `extern "C"`, no libc, no BLAS, no llama.cpp.

The name reflects the goal: the engine's overhead should *disappear*. No
interpreter tax, no copy tax, no allocation tax — just weights and math.

---

## Status

| Capability                              | State                |
|-----------------------------------------|----------------------|
| GGUF v2/v3 parsing (zero-copy mmap)     | shipped              |
| Llama-architecture forward pass         | shipped              |
| KV cache (fixed-size, layer-major)      | shipped              |
| Persistent thread pool (N-1 workers)    | shipped              |
| SentencePiece BPE encode + decode       | shipped              |
| Sampling: greedy / temp / top-k / top-p | shipped              |
| Chat template (TinyLlama zephyr)        | shipped              |
| HTTP server (Ollama-compatible subset)  | shipped              |
| Apple Metal GPU backend (Q8_0 matmul)   | shipped (opt-in `-Dmetal=true`) |
| Q8_0 / Q4_K / Q5_K / Q6_K / F32 / F16   | shipped              |
| TQ2_0 (1.58-bit ternary) plumbed        | dequant + matmul written, untested on real model |
| Safetensors loader                      | not yet              |
| GPU (Metal / Vulkan)                    | not yet (see *Pure Zig* note below) |
| Prompt prefill batching                 | not yet              |

**Verified end-to-end** on TinyLlama-1.1B-Chat-v1.0 GGUF Q8_0, Q4_K_M, Q4_K_S
(ReleaseFast, macOS arm64). Coherent text generation, canonical SentencePiece
tokenization, EOS-correct chat-template completions.

---

## Performance

TinyLlama-1.1B-Chat-v1.0 on macOS arm64 (M-series), ReleaseFast, 30-token
generation, warm caches:

| Build                          | CPU%   | tok/s | Wall  | Notes                                    |
|--------------------------------|--------|-------|-------|------------------------------------------|
| CPU `--threads 16`             | 1529%  | 45    | 0.67s | All cores pinned; max throughput         |
| CPU default (~8)               | 739%   | 28    | 1.07s | Half cores; default                      |
| **Metal default (1 CPU thread)** | **27%**  | **23**| **1.48s** | **Q8_0 matmul on GPU; near-idle CPU** |
| Metal + 8 CPU threads          | 672%   | 17    | 1.74s | CPU spins on GPU sync — don't do this     |

The headline win: **`-Dmetal=true` + default thread count drops CPU usage
to under 30%** while still hitting 23 tok/s on Q8_0. Fan stays off,
laptop stays cool, foreground work stays responsive.

Q8_0 matmul runs on the GPU; smaller ops (RMSNorm, RoPE, softmax,
SwiGLU, attention) and other quants stay on the CPU. With Metal enabled
the default is `--threads 1` because additional CPU workers just spin on
GPU sync.

For comparison, llama.cpp on the same hardware does ~50–100 tok/s on
Q8_0 (uses its own Metal compute path with all ops on GPU + bigger
threadgroup tiles). vantablack's Metal path is a first cut — Q8_0 only,
naive scalar inner loop, sync per matmul. Roadmap below covers the
remaining 2–3× perf headroom.

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
                                  kernels/
                                  comptime_gen
                                  (dispatch by quant)
                                       |
                                       v
                          +------------+--------+
                          | matmul_q8_0         |
                          | matmul_q4_k         |
                          | matmul_q5_k         |
                          | matmul_q6_k         |
                          | matmul_f16/f32      |
                          | matmul_ternary158   |
                          +---------------------+
```

### Module map

```
src/
├── main.zig                  CLI entry point + subcommand dispatch
├── vantablack.zig            public module index (re-exports)
├── core/
│   ├── mapper.zig            ModelMapper: open + std.posix.mmap + catalog
│   └── parser.zig            GGUF v2/v3 parser, MetaValue union, BlockInfo table
├── kernels/
│   ├── simd.zig              real Q8_0/Q4_K/Q5_K/Q6_K/TQ2_0/F16/F32 dequant + matmul
│   ├── math.zig              RMSNorm / RoPE / softmax / SwiGLU / argmax
│   └── comptime_gen.zig      QuantType + Kernel + dispatch(comptime q)
└── runtime/
    ├── kv_cache.zig          fixed-size O(1)-memory KV cache
    ├── model.zig             LlamaConfig + Model.init from GGUF metadata
    ├── forward.zig           per-token forward pass; matmul row-split via pool
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

Spec rule #5 calls for a unified-memory GPU path via Metal and Vulkan.
Both APIs require FFI (Objective-C runtime for Metal, C ABI for Vulkan/MoltenVK),
which conflicts with the no-`extern "C"` constraint. Resolving this requires
either waiving the rule or shipping raw Mach-O `dlsym`-based bindings. Not yet
attempted.

---

## Build

Requires Zig 0.16.0+.

```sh
zig build                                      # Debug, default
zig build -Doptimize=ReleaseFast               # multi-threaded CPU, fast inference
zig build -Doptimize=ReleaseFast -Dmetal=true  # CPU + Apple Metal GPU (Q8_0 matmul on GPU)
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
   `-Dmetal=true` build flag. Q8_0 matmuls run on the GPU via a small
   Obj-C bridge (`src/metal/bridge.m`) and an inline MSL kernel. Weights
   are wrapped as zero-copy `MTLBuffer`s over the existing mmap region
   (Apple Silicon unified memory makes this free). Default thread count
   drops to 1 when Metal is active. **CPU usage falls to ~27% at 23 tok/s**
   on TinyLlama Q8_0 — fan stays off.

   Trade-off: this required relaxing the original "pure Zig std only"
   rule. The Metal backend links Foundation + Metal frameworks and
   compiles ~250 LoC of Objective-C. The pure-Zig CPU path is preserved
   and remains the default when `-Dmetal=true` is not passed.

4. **All ops on GPU (RMSNorm, attention, RoPE, softmax, SwiGLU)** —
   *planned*. Currently we sync GPU↔CPU per matmul (~220 syncs per
   token). Moving the small ops to MSL too would let one command buffer
   process a full layer. Would also enable Q4_K/Q5_K/Q6_K kernels.
5. **Tighter MSL matmul (threadgroup tiling, simdgroup matmul)** —
   *planned*. Current kernel is one thread per output row with scalar
   inner loop. M-series GPUs expose `simdgroup_multiply_accumulate` for
   8×8 tile FMA; using it gives ~4× kernel speedup.
6. **Apple AMX matrix coprocessor** — *planned*. M1/M2/M3 ship a hidden
   matrix unit accessible via inline assembly. Pure-Zig + lower power
   than even GPU for some shapes. Useful as a CPU-only fallback path.
7. **NEON `sdot` int8 dot** — *planned*. ARMv8.2-A dedicated dot-product
   instruction. Modest CPU-path speedup, no FFI needed.
8. **Prompt prefill batching** — *planned*. Process N prompt tokens in
   one forward pass. Big speedup on long prompts.

Items 4 + 5 are the next big perf push: `-Dmetal=true` should hit
50–100+ tok/s like llama.cpp once attention + small ops also run on the
GPU.

---

## Limitations + future work

- **Cross-thread floating-point non-determinism.** Matmul partial sums are
  reduced per-worker, then summed. fp32 addition is non-associative, so
  greedy outputs can differ slightly between runs (or vs single-threaded).
  Outputs stay coherent.
- **No prompt prefill batching.** Prompt tokens are processed one at a time.
  For an N-token prompt the cost is N forward passes. Real throughput on
  long prompts will improve substantially with batched prefill.
- **TQ2_0 untested on real weights.** Spec-derived layout, internally
  consistent, no public TQ2_0 GGUF weights to validate against.
- **Single-architecture (Llama).** Mistral / Phi / Gemma metadata key
  remapping not done.
- **No tokenizer encoder fallback.** SentencePiece BPE only. Tiktoken-style
  byte-level BPE (GPT-2/Llama-3) requires a different merge algorithm.
- **GPU partial coverage.** `-Dmetal=true` ships Q8_0 matmul on GPU but
  attention / norms / softmax / SwiGLU still run on CPU. One sync per
  matmul (~220 per token). Throughput is ~23 tok/s vs ~45 tok/s on full
  CPU pin — but at 27% CPU instead of 1500%. Roadmap items 4+5 in "CPU,
  power, GPU" close this gap.
- **Yielding workers, not parking.** Idle workers `std.Thread.yield()`
  rather than blocking on a futex/condvar — Zig 0.16's std doesn't expose
  futex publicly without libc. Idle CPU drops sharply but is not zero.
- **Server is serial.** One request at a time, mutex-guarded. Good for
  single-user code-gen; multi-tenant requires multiple `serve` processes
  on different ports.
- **Server `created_at` is a placeholder** (`1970-01-01T00:00:00Z`) and
  `digest`/`total_duration`/`eval_count` fields are omitted. Most clients
  tolerate it; full Ollama parity would add real timestamps + sha256.

---

## Build history

The engine grew in five tracked phases. Commit log mirrors this:

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

Each phase preserved the hard constraints that still apply (mmap-first,
< 5 MB stripped, no global state) and shipped passing tests + real-model
smoke before moving on.

---

## License

See `LICENSE`.
