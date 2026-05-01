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
generation, single CLI invocation (warm caches, no prefill batching):

| Quant     | Tokens/sec  | CPU%       |
|-----------|-------------|------------|
| Q8_0      | **31**      | 1347%      |
| Q4_K_S    | **17.6**    | 1458%      |
| Q4_K_M    | **25**      | 1756%      |

For comparison, llama.cpp on the same hardware does ~50–100 tok/s on Q8_0.
Headroom: NEON-tuned int8 dot path, prompt prefill batching, M2/M3 AMX dispatch.

---

## Architecture

```
                     argv: model.gguf [subcommand args]
                                  |
                                  v
        +-------------------------+--------------------------+
        |                  src/main.zig CLI                  |
        |   (catalog | generate | prompt | chat dispatch)    |
        +------+----------+-----------+---------+------------+
               |          |           |         |
               v          v           v         v
        ModelMapper   Tokenizer    Sampler  ThreadPool
       (mmap+catalog)  (BPE+SPM)  (temp/k/p) (N-1 workers)
               |          |           |         |
               +-----+----+-----+-----+----+----+
                     |          |          |
                     v          v          v
                +----+----------+----------+----+
                |       runtime/forward.zig       |
                |  per-token Llama forward pass  |
                |  (RMSNorm, QKV, RoPE, GQA-attn,|
                |   SwiGLU FFN, residuals,       |
                |   LM head -> logits)           |
                +----+--------+--------+---------+
                     |        |        |
                     v        v        v
                  KvCache   math/    kernels/
                            simd     comptime_gen
                                   (dispatch by quant)
                                          |
                                          v
                              +-----------+--------+
                              | matmul_q8_0        |
                              | matmul_q4_k        |
                              | matmul_q5_k        |
                              | matmul_q6_k        |
                              | matmul_f16/f32     |
                              | matmul_ternary158  |
                              +--------------------+
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
    └── sampler.zig           temp/top-k/top-p + std.Random PRNG
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
   217 KB. Pulling in a dependency would feel as expensive as it actually is.

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
zig build                         # Debug, default
zig build -Doptimize=ReleaseFast  # multi-threaded, fast inference
zig build -Doptimize=ReleaseSmall # single-threaded, ~217 KB stripped
zig build test                    # all unit tests
```

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
```

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
- **CPU-only.** GPU path conflicts with no-FFI constraint.

---

## License

See `LICENSE`.
