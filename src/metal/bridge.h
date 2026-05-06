// Flat C API bridge to Apple Metal. Compiled into the executable when
// `-Dmetal=true` is passed to the build. Zig consumers use `extern "c"`
// declarations against these symbols.
//
// Lifetime: VtbMetalCtx is process-long; create at startup via
// vtb_metal_init, destroy at shutdown. Buffers (VtbMetalBuf) are
// reference-counted under ARC; vtb_metal_release drops the bridge's
// reference.

#ifndef VTB_METAL_BRIDGE_H
#define VTB_METAL_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VtbMetalCtx VtbMetalCtx;
typedef struct VtbMetalBuf VtbMetalBuf;

// Initialize the Metal context: device, queue, library, pipelines.
// Returns NULL if no compatible device is available or init failed.
VtbMetalCtx *vtb_metal_init(void);
void vtb_metal_destroy(VtbMetalCtx *ctx);

// Allocate a shared-storage MTLBuffer of `bytes`. Returns the CPU-accessible
// pointer to its contents in *ptr_out. Returns NULL on failure.
VtbMetalBuf *vtb_metal_alloc(VtbMetalCtx *ctx, size_t bytes, void **ptr_out);

// Wrap an existing host pointer as a shared MTLBuffer with no copy. `ptr`
// must be page-aligned (mmap'd memory qualifies). `len` must be a multiple
// of the page size. Returns NULL on failure.
VtbMetalBuf *vtb_metal_wrap(VtbMetalCtx *ctx, const void *ptr, size_t len);

// Drop the bridge's reference to the buffer. Underlying memory is freed
// when no more references remain.
void vtb_metal_release(VtbMetalBuf *buf);

// Q8_0 matmul: out = W @ acts. Synchronous; blocks until GPU completes.
//   out_buf:        VtbMetalBuf of m * sizeof(float) — result
//   w_buf:          VtbMetalBuf containing the weight bytes
//   w_offset:       byte offset within w_buf where this tensor's row 0 starts
//   acts_buf:       VtbMetalBuf of k * sizeof(float) — input activations
//   m, k:           dimensions (out is len m; acts is len k; W is m rows of k cols Q8_0)
// Returns 0 on success, nonzero on failure.
int vtb_metal_matmul_q8_0(
    VtbMetalCtx *ctx,
    VtbMetalBuf *out_buf,
    VtbMetalBuf *w_buf,
    size_t w_offset,
    VtbMetalBuf *acts_buf,
    size_t m,
    size_t k);

// Q4_K / Q5_K / Q6_K matmul. Same signature as Q8_0; block layouts match
// `src/kernels/simd.zig::dequantBlockQ{4,5,6}_K`. Synchronous one-shot.
int vtb_metal_matmul_q4_k(
    VtbMetalCtx *ctx, VtbMetalBuf *out_buf,
    VtbMetalBuf *w_buf, size_t w_offset,
    VtbMetalBuf *acts_buf, size_t m, size_t k);
int vtb_metal_matmul_q5_k(
    VtbMetalCtx *ctx, VtbMetalBuf *out_buf,
    VtbMetalBuf *w_buf, size_t w_offset,
    VtbMetalBuf *acts_buf, size_t m, size_t k);
int vtb_metal_matmul_q6_k(
    VtbMetalCtx *ctx, VtbMetalBuf *out_buf,
    VtbMetalBuf *w_buf, size_t w_offset,
    VtbMetalBuf *acts_buf, size_t m, size_t k);

// ----- Batched / fused dispatch -------------------------------------------
//
// `vtb_metal_segment_*` lets the caller chain multiple operations into one
// MTLCommandBuffer and flush only when CPU needs to read intermediate
// results. Cuts per-token GPU↔CPU sync count dramatically.

typedef struct VtbMetalSeg VtbMetalSeg;

VtbMetalSeg *vtb_metal_segment_begin(VtbMetalCtx *ctx);
// Commits the segment without waiting. Caller must invoke vtb_metal_wait_idle
// before reading any shared-storage buffer the segment wrote. Returns 0 on
// success.
int vtb_metal_segment_commit(VtbMetalSeg *seg);

// Drain the serial command queue. Blocks until every command buffer enqueued
// before this call has finished executing. Required before the CPU reads
// shared-storage buffers written by async segments.
void vtb_metal_wait_idle(VtbMetalCtx *ctx);

// All segment_* fns enqueue into `seg`. Buffer dimensions must already match.

void vtb_metal_segment_matmul_q8_0(
    VtbMetalSeg *seg,
    VtbMetalBuf *out_buf,
    VtbMetalBuf *w_buf,
    size_t w_offset,
    VtbMetalBuf *acts_buf,
    size_t m,
    size_t k);

void vtb_metal_segment_matmul_q4_k(
    VtbMetalSeg *seg, VtbMetalBuf *out_buf,
    VtbMetalBuf *w_buf, size_t w_offset,
    VtbMetalBuf *acts_buf, size_t m, size_t k);
void vtb_metal_segment_matmul_q5_k(
    VtbMetalSeg *seg, VtbMetalBuf *out_buf,
    VtbMetalBuf *w_buf, size_t w_offset,
    VtbMetalBuf *acts_buf, size_t m, size_t k);
void vtb_metal_segment_matmul_q6_k(
    VtbMetalSeg *seg, VtbMetalBuf *out_buf,
    VtbMetalBuf *w_buf, size_t w_offset,
    VtbMetalBuf *acts_buf, size_t m, size_t k);

void vtb_metal_segment_rmsnorm(
    VtbMetalSeg *seg,
    VtbMetalBuf *out_buf,
    VtbMetalBuf *in_buf,
    VtbMetalBuf *weight_buf,
    size_t weight_offset,
    size_t n,
    float eps);

void vtb_metal_segment_rope(
    VtbMetalSeg *seg,
    VtbMetalBuf *x_buf,
    size_t n_heads,
    size_t head_dim,
    size_t pos,
    float base);

void vtb_metal_segment_swiglu(
    VtbMetalSeg *seg,
    VtbMetalBuf *gate_buf,
    VtbMetalBuf *up_buf,
    size_t n);

// Gemma activation: gate[i] = gelu_approx(gate[i]) * up[i] in place.
void vtb_metal_segment_gelu_approx(
    VtbMetalSeg *seg,
    VtbMetalBuf *gate_buf,
    VtbMetalBuf *up_buf,
    size_t n);

void vtb_metal_segment_residual_add(
    VtbMetalSeg *seg,
    VtbMetalBuf *a_buf,
    VtbMetalBuf *b_buf,
    size_t n);

// f32 copy: dst[dst_off + i] = src[src_off + i] for i in [0, n_floats).
// Offsets are byte offsets, applied via setBuffer:offset: so they must be
// 4-byte (f32) aligned.
void vtb_metal_segment_copy(
    VtbMetalSeg *seg,
    VtbMetalBuf *dst_buf, size_t dst_offset_bytes,
    VtbMetalBuf *src_buf, size_t src_offset_bytes,
    size_t n_floats);

// scores[h, t] = (q[h] · K[t, kv_h]) * inv_sqrt_hd, for h ∈ n_heads,
// t ∈ seq_len. K cache offset (bytes) selects the per-layer KV slab.
void vtb_metal_segment_attn_scores(
    VtbMetalSeg *seg,
    VtbMetalBuf *scores_buf,
    VtbMetalBuf *q_buf,
    VtbMetalBuf *k_cache_buf, size_t k_cache_offset_bytes,
    size_t n_heads, size_t n_kv_heads, size_t head_dim, size_t seq_len,
    float inv_sqrt_hd);

// In-place row-wise numerically-stable softmax. One row per head, length
// seq_len. Stride between rows is exactly seq_len (not max_seq).
void vtb_metal_segment_softmax_rows(
    VtbMetalSeg *seg,
    VtbMetalBuf *scores_buf,
    size_t n_heads, size_t n_kv_heads, size_t head_dim, size_t seq_len);

// out[h, d] = Σ_t scores[h, t] * V[t, kv_h, d].
void vtb_metal_segment_attn_weighted_sum(
    VtbMetalSeg *seg,
    VtbMetalBuf *out_buf,
    VtbMetalBuf *scores_buf,
    VtbMetalBuf *v_cache_buf, size_t v_cache_offset_bytes,
    size_t n_heads, size_t n_kv_heads, size_t head_dim, size_t seq_len);

// MLX-format 4-bit affine block-quant matmul. Weight, scales, and biases
// may live in different MTLBuffers (different safetensors shards), so the
// dispatch takes one buffer + byte offset per tensor.
//
//   out_buf:           shared-storage scratch, m floats
//   weight_buf,off:    packed U32 nibbles, m * (k/8) u32s (m * k/2 bytes)
//   scales_buf,off:    F16 or BF16 scales, m * (k/group_size) entries
//   biases_buf,off:    same layout as scales, F16 or BF16
//   acts_buf,off:      k floats
//   m, k:              dims
//   group_size:        quant group (typically 64)
//   scale_dtype_is_bf16: 0 = f16, 1 = bf16
//
// Metal compute encoders require setBuffer:offset: to be 4-byte aligned, but
// safetensors data sections begin at `8 + json_header_len` and the JSON
// header length is rarely a multiple of 4 — so absolute tensor offsets are
// commonly 1 or 3 mod 4. The wrapper snaps each offset down to the nearest
// 4-byte boundary (`& ~3`) for the bind, then passes the 0..3-byte residual
// through MlxQ4Params.{w,scales,biases}_offset_bytes; the kernel adds the
// residual back in via byte-wise pointer arithmetic. The acts buffer is f32
// scratch and is always 4-byte aligned, so it's bound at its raw offset.
void vtb_metal_segment_matmul_mlx_q4(
    VtbMetalSeg *seg,
    VtbMetalBuf *out_buf,
    VtbMetalBuf *weight_buf, size_t weight_offset,
    VtbMetalBuf *scales_buf, size_t scales_offset,
    VtbMetalBuf *biases_buf, size_t biases_offset,
    VtbMetalBuf *acts_buf,   size_t acts_offset,
    size_t m, size_t k,
    uint32_t group_size,
    uint32_t scale_dtype_is_bf16);

// One-shot variant of vtb_metal_segment_matmul_mlx_q4: builds, commits, and
// waits on its own command buffer. Returns 0 on success, nonzero on failure.
int vtb_metal_matmul_mlx_q4(
    VtbMetalCtx *ctx,
    VtbMetalBuf *out_buf,
    VtbMetalBuf *weight_buf, size_t weight_offset,
    VtbMetalBuf *scales_buf, size_t scales_offset,
    VtbMetalBuf *biases_buf, size_t biases_offset,
    VtbMetalBuf *acts_buf,   size_t acts_offset,
    size_t m, size_t k,
    uint32_t group_size,
    uint32_t scale_dtype_is_bf16);

#ifdef __cplusplus
}
#endif

#endif // VTB_METAL_BRIDGE_H
