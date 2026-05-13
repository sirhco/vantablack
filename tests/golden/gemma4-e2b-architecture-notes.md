# Gemma 4 E2B architecture (reverse-engineered via `vantablack inspect-section`)

Source: `litert-community/gemma-4-E2B-it-litert-lm` (2.6 GB `.litertlm`).
Captured via deep dump of section 10 (780 MB, the decoder block).

## Bundle layout

| Section | Type | Size | Role |
|---------|------|------|------|
| 0 | LlmMetadataProto | 12 KB | Gemma4 type + 11995-char Jinja chat template + 3 stop tokens. No sampler_params. |
| 1 | SP_Tokenizer | 4.5 MB | SentencePiece. |
| 2 | TFLiteModel | 99 MB | Text embedder lookup (INT2). |
| 3 | TFLiteModel | 1225 MB | Per-layer embedder (35× INT4 `[262144, 256]` shards). |
| 4 | TFLiteModel | 90 MB | **AudioConformerEncoder** (multimodal — audio input). |
| 5 | TFLiteModel | 9 MB | **AudioAdapter** (projects audio embed into text space). |
| 6 | TFLiteModel | 7 KB | Initial-state stub. |
| 7 | TFLiteModel | 214 MB | Vision adapter (suspected). |
| 8 | TFLiteModel | 5 MB | Small block. |
| 9 | TFLiteModel | 7 KB | Initial-state stub. |
| **10** | **TFLiteModel** | **780 MB** | **`LanguageModel.decode_graph` — the transformer decoder.** |
| 11 | TFLiteModel | 42 MB | `LanguageModel.decode_softmax` + LM head + final SOFTMAX. |

## Architecture (from section 10 tensor names)

- **Hidden dim**: 1536
- **FFN inner dim**: 12288 (8× hidden — wider than standard Llama 4× ratio)
- **Layer count**: ≥ 35 (highest observed: `layer_34`)
- **Vocab**: 262144
- **MLP**: SwiGLU 3-matrix style
  - `gating_einsum1` (gate proj): `[B,T,D] @ [D,F] → [B,T,F]`  shape=`[12288, 1536]`
  - `gating_einsum2` (up proj): `[B,T,D] @ [D,F] → [B,T,F]`    shape=`[12288, 1536]`
  - `linear` (down proj): `[B,T,F] @ [F,D] → [B,T,D]`          shape=`[1536, 12288]`
- **Attention**: not yet traced — tensor names use `post_qkv` suggesting Q/K/V are already split before the matmul; tracing input/output graph required.

## Quantization breakdown (section 10)

- Most MLP weights: **INT2** (extreme compression)
- Some weights (layer 2 `gating_einsum2`): **INT4**
- Linear projections in section 11: **INT8** (`per_layer_model_projection`)
- Some activations: **INT8** with per-axis scale arrays
- Norms / scales: **FLOAT32** (RMSNorm weights presumably)

## Tensor naming convention

```
LanguageModel.decode_graph/LanguageModel.transformer_stack/transformer.transformer/layer_{N}/layer_{N}.post_qkv/mlp/{gating_einsum1|gating_einsum2|linear}/btd,df->btf/dot_general
```

Per-layer pattern is consistent. Phase 19c name-pattern mapping is now
tractable — split on `/`, parse `layer_{N}`, map the `{gate|up|down}` slot
by einsum signature or trailing `gating_einsum1|2|linear`.

## Section 11 (LM head + sampling)

- 275 tensors, 198 operators
- Op codes include FULLY_CONNECTED, SOFTMAX, TANH, MIRROR_PAD
- Suspected: final RMSNorm + LM head matmul + SOFTMAX + token emit
- Confirms 19f hypothesis — sampling happens inside this section's TFLite
  graph, not via a host-side sampler.

## Per-layer scan (added after running `scan-layers`)

`vantablack scan-layers <model.litertlm>` (output saved to
`gemma4-e2b-scan-layers.txt`) classifies all 35 decoder layers with
complete coverage on 5 roles:

| Role | Coverage | Layer-0 example shape | Layer-0 dtype |
|------|----------|-----------------------|---------------|
| mlp.gate (gating_einsum1) | 35/35 | [6144, 1536] | INT4 |
| mlp.up   (gating_einsum2) | 35/35 | [6144, 1536] | INT4 |
| mlp.down (linear)         | 35/35 | [1536, 6144] | INT4 |
| attn.o (post_qkv/attn)    | 35/35 | [1536, 2048] | INT4 |
| qkv (pre_qkv/attn)        | 35/35 | [2048, 1536] | INT4 |
| ple.gate                  | 35/35 | [256, 1536]  | INT8 |
| ple.proj                  | 35/35 | [1536, 256]  | INT8 |

Observations:

- **FFN dim varies by layer**: layer 0 has F=6144 (4× hidden); deeper
  layers have F=12288 (8× hidden). MatFormer-style variable width.
- **Quant varies by layer**: layer 0 is INT4; many deeper layers are
  INT2. Per-layer quantization decision baked into the build.
- **QKV shape [2048, 1536] is suspicious**: 2048 doesn't fit a single
  combined Q+K+V projection (would be n_q_heads*head_dim +
  2*n_kv_heads*head_dim, typically larger). May only be the Q
  projection — K/V might live elsewhere (separate tensors). Needs
  op-graph trace to confirm.
- **Norms classified as "unknown"**: 5 norms per layer (pre/post
  attention/ffw + per-layer-input) — `scan-layers` patterns don't yet
  match all of them. Refinement task.

## What this unlocks

- **19c**: write a tensor-name pattern matcher. Output: a vantablack
  `LayerWeights` populated from section 10's `layer_{N}` tensors.
  **Foundation shipped** as `core/gemma_layer_scan.zig` + `scan-layers`
  CLI. Remaining: refine norm patterns, confirm qkv shape via op
  graph, then wire to a new `Model` variant (Gemma 4 isn't Llama-
  compatible — has PLE + 5 norms).
- **19d**: INT2 + INT4 + INT8 dequant kernels. INT2 is novel — needs a
  bit-packed scale/zero-point layout described in TFLite's
  `QuantizationParameters.scale[] + zero_point[]` per-axis.
- **19f**: parse + execute section 11's SOFTMAX + sampling chain. Inputs
  come from the decoder's last hidden state; output is the next token id.

## Next concrete steps

1. Add `inspect-section --filter <regex>` so callers can grep tensor names.
2. Trace one full layer's op graph (e.g. layer_0) — list every operator
   whose inputs/outputs reference a `layer_0/*` tensor — to confirm the
   forward path matches Llama.
3. Implement INT2 dequant on CPU (no kernel exists today, this is novel).
