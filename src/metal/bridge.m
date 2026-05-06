// Apple Metal GPU bridge for vantablack.
//
// Compiled with -fobjc-arc by build.zig. Owns a single MTLDevice +
// MTLCommandQueue + MTLLibrary + cached MTLComputePipelineState per kernel.
// The MSL kernel source is embedded as a string literal so we don't need a
// separate .metallib file or runtime loader.
//
// All buffers use MTLResourceStorageModeShared so the CPU and GPU see the
// same physical memory on Apple Silicon — no DMA, no copy.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "bridge.h"

// ----------------- MSL kernels -----------------------------------------------

static NSString *const kMslSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint Q8_BLOCK_ELEMS = 32;
constant uint Q8_BLOCK_BYTES = 34;

struct MMParams {
    uint m;
    uint k;
    uint w_offset_bytes; // byte offset into the weights buffer
};

// Q8_0 matmul, per-thread per-row.
//
// One thread = one output row. Inside each row, the 32-element Q8_0 block
// loop is unrolled into 8 float4 FMAs. Apple's L2 cache absorbs the activation
// re-reads across simdgroups, and the dot-product chain stays in registers,
// so this simple layout already saturates the memory subsystem for TinyLlama-
// scale matmuls. Empirically beats both shared-tile and simdgroup-per-row
// variants on M-series — see project memory.

kernel void matmul_q8_0(
    device float       *out      [[buffer(0)]],
    device const uchar *weights  [[buffer(1)]],
    device const float *acts     [[buffer(2)]],
    constant MMParams  &p        [[buffer(3)]],
    uint                gid      [[thread_position_in_grid]])
{
    if (gid >= p.m) return;
    uint blocks_per_row = p.k / Q8_BLOCK_ELEMS;
    uint row_bytes = blocks_per_row * Q8_BLOCK_BYTES;
    device const uchar *row = weights + p.w_offset_bytes + gid * row_bytes;

    float total = 0.0;
    for (uint b = 0; b < blocks_per_row; ++b) {
        device const uchar *blk = row + b * Q8_BLOCK_BYTES;
        ushort scale_u16 = (ushort)blk[0] | ((ushort)blk[1] << 8);
        float scale_f = (float)as_type<half>(scale_u16);

        // qs lives at byte offset 2 within the block — only 2-byte aligned,
        // so packed_char4 (no alignment requirement) is required.
        device const packed_char4 *qs4 = (device const packed_char4 *)(blk + 2);
        device const float4 *x4 = (device const float4 *)(acts + b * Q8_BLOCK_ELEMS);

        float4 acc = float4(0.0);
        acc = fma(float4(char4(qs4[0])), x4[0], acc);
        acc = fma(float4(char4(qs4[1])), x4[1], acc);
        acc = fma(float4(char4(qs4[2])), x4[2], acc);
        acc = fma(float4(char4(qs4[3])), x4[3], acc);
        acc = fma(float4(char4(qs4[4])), x4[4], acc);
        acc = fma(float4(char4(qs4[5])), x4[5], acc);
        acc = fma(float4(char4(qs4[6])), x4[6], acc);
        acc = fma(float4(char4(qs4[7])), x4[7], acc);
        total = fma(scale_f, acc.x + acc.y + acc.z + acc.w, total);
    }
    out[gid] = total;
}

// In-place RMSNorm: x[i] = (x[i] / rms(x)) * weight[i]
// One threadgroup per call. Threads collaboratively compute sum-of-squares
// using simdgroup reduction, then apply the normalization.

struct RmsNormParams {
    uint n;
    float eps;
};

// Two-buffer RMSNorm: out[i] = (in[i] / rms(in)) * weight[i].
// Lets the caller chain residual_add(x, …) → rmsnorm(out=xb, in=x) without a
// memcpy in between. Pass in == out for in-place.
kernel void rmsnorm(
    device float        *out     [[buffer(0)]],
    device const float  *in      [[buffer(1)]],
    device const float  *weight  [[buffer(2)]],
    constant RmsNormParams &p    [[buffer(3)]],
    uint                 tid     [[thread_position_in_threadgroup]],
    uint                 ntg     [[threads_per_threadgroup]])
{
    float ss = 0.0;
    for (uint i = tid; i < p.n; i += ntg) {
        float v = in[i];
        ss += v * v;
    }
    threadgroup float scratch[32];
    uint sid = tid % 32;
    uint sgrp = tid / 32;
    ss = simd_sum(ss);
    if (sid == 0) scratch[sgrp] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgrp == 0) {
        ss = (sid < (ntg + 31) / 32) ? scratch[sid] : 0.0;
        ss = simd_sum(ss);
        if (sid == 0) scratch[0] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_ss = scratch[0];
    float inv = rsqrt(total_ss / float(p.n) + p.eps);
    for (uint i = tid; i < p.n; i += ntg) {
        out[i] = in[i] * inv * weight[i];
    }
}

// In-place RoPE on Q or K. Each kernel invocation handles all heads/pairs.
// One thread per (head, pair).

struct RopeParams {
    uint n_heads;
    uint head_dim;
    uint pos;
    float base;
};

kernel void rope_inplace(
    device float        *x       [[buffer(0)]],
    constant RopeParams &p       [[buffer(1)]],
    uint                 gid     [[thread_position_in_grid]])
{
    uint pairs_per_head = p.head_dim / 2;
    uint total_pairs = p.n_heads * pairs_per_head;
    if (gid >= total_pairs) return;
    uint h = gid / pairs_per_head;
    uint pair_idx = gid % pairs_per_head;
    uint i = pair_idx * 2;
    float inv_freq = 1.0 / pow(p.base, float(i) / float(p.head_dim));
    float theta = float(p.pos) * inv_freq;
    float c = cos(theta);
    float s = sin(theta);
    uint off = h * p.head_dim + i;
    float x0 = x[off];
    float x1 = x[off + 1];
    x[off]     = x0 * c - x1 * s;
    x[off + 1] = x0 * s + x1 * c;
}

// SwiGLU: gate[i] = silu(gate[i]) * up[i] where silu(z) = z / (1 + exp(-z)).

struct LenParams { uint n; };

kernel void swiglu(
    device float        *gate    [[buffer(0)]],
    device const float  *up      [[buffer(1)]],
    constant LenParams  &p       [[buffer(2)]],
    uint                 gid     [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    float z = gate[gid];
    float silu = z / (1.0 + exp(-z));
    gate[gid] = silu * up[gid];
}

// GeGLU with tanh-approximate GELU (Gemma): gate[i] = gelu_approx(gate[i]) * up[i].
// gelu_approx(z) = 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))
// — matches the math.zig::gegluApprox CPU reference.
kernel void gelu_approx(
    device float        *gate    [[buffer(0)]],
    device const float  *up      [[buffer(1)]],
    constant LenParams  &p       [[buffer(2)]],
    uint                 gid     [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    float z = gate[gid];
    float inner = 0.7978845608028654f * (z + 0.044715f * z * z * z);
    float gelu = 0.5f * z * (1.0f + tanh(inner));
    gate[gid] = gelu * up[gid];
}

// In-place residual: a[i] += b[i].

kernel void residual_add(
    device float        *a       [[buffer(0)]],
    device const float  *b       [[buffer(1)]],
    constant LenParams  &p       [[buffer(2)]],
    uint                 gid     [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    a[gid] = a[gid] + b[gid];
}

// MLX 4-bit affine block-quant matmul (GEMV). Mirrors the layout produced
// by `mlx_lm.utils.quantize` with bits=4, group_size ∈ {64, 128}. Three
// device buffers per layer: packed weights (U32, 8 nibbles each), scales
// and biases (F16 or BF16, one entry per group). One thread per output row.
// Dequant happens implicitly inside the inner loop:
//   total = Σ_g (scale_g * Σ_{c in group_g} q[c] * x[c]
//             + bias_g  * Σ_{c in group_g}        x[c])
//
// `scale_dtype_is_bf16` lets the same kernel cover both BF16 (modern MLX)
// and F16 (legacy) scales/biases.

struct MlxQ4Params {
    uint m;
    uint k;
    uint w_offset_bytes;        // packed weight offset
    uint scales_offset_bytes;
    uint biases_offset_bytes;
    uint group_size;
    uint scale_dtype_is_bf16;   // 0 = f16, 1 = bf16
};

inline float bf16_to_f32(ushort bits) {
    uint w = ((uint)bits) << 16;
    return as_type<float>(w);
}

// Byte-wise unaligned u16 load. The MLX-Q4 host helper baked the per-row
// pointer arithmetic into the kernel via `*_offset_bytes`, so the resulting
// address is only guaranteed 1-byte-aligned (safetensors stores the data
// section at `8 + json_header_len`, and the JSON header length is rarely a
// multiple of 4). A direct `(device const ushort *)` cast on a misaligned
// pointer returns garbage on Apple GPUs.
// LE byte order: matches safetensors + Apple GPU native order.
inline ushort load_u16_unaligned(device const uchar *p) {
    return (ushort)p[0] | ((ushort)p[1] << 8);
}

// LE byte order: matches safetensors + Apple GPU native order.
inline uint load_u32_unaligned(device const uchar *p) {
    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

inline float scale_load_bytes(device const uchar *base, uint byte_idx, bool is_bf16) {
    ushort bits = load_u16_unaligned(base + byte_idx);
    if (is_bf16) return bf16_to_f32(bits);
    half h = as_type<half>(bits);
    return (float)h;
}

kernel void matmul_mlx_q4(
    device float          *out     [[buffer(0)]],
    device const uchar    *weights [[buffer(1)]],
    device const uchar    *scales  [[buffer(2)]],
    device const uchar    *biases  [[buffer(3)]],
    device const float    *acts    [[buffer(4)]],
    constant MlxQ4Params  &p       [[buffer(5)]],
    uint                   gid     [[thread_position_in_grid]])
{
    if (gid >= p.m) return;

    bool is_bf16 = (p.scale_dtype_is_bf16 != 0);
    uint groups_per_row = p.k / p.group_size;
    uint u32_per_row = p.k / 8;            // 8 nibbles per u32 for 4-bit
    uint weight_row_bytes = u32_per_row * 4;
    uint scale_row_bytes = groups_per_row * 2;

    device const uchar *w_row = weights + p.w_offset_bytes + gid * weight_row_bytes;
    device const uchar *s_row = scales + p.scales_offset_bytes + gid * scale_row_bytes;
    device const uchar *b_row = biases + p.biases_offset_bytes + gid * scale_row_bytes;

    float total = 0.0;
    uint col = 0;
    for (uint g = 0; g < groups_per_row; ++g) {
        float scale = scale_load_bytes(s_row, g * 2, is_bf16);
        float bias = scale_load_bytes(b_row, g * 2, is_bf16);
        float q_dot = 0.0;
        float x_sum = 0.0;
        uint end = col + p.group_size;
        for (uint c = col; c < end; c += 8) {
            // Byte-wise u32 load: w_row may be 1-byte-aligned (safetensors
            // data section starts mid-page on real bundles), so a direct
            // `device const uint *` cast can return garbage on AGX hardware.
            uint word = load_u32_unaligned(w_row + (c / 8) * 4);
            for (uint i = 0; i < 8; ++i) {
                uint q = (word >> (i * 4)) & 0xF;
                float xv = acts[c + i];
                q_dot = fma((float)q, xv, q_dot);
                x_sum += xv;
            }
        }
        total = fma(scale, q_dot, total);
        total = fma(bias, x_sum, total);
        col = end;
    }
    out[gid] = total;
}

// ----- K-quant matmul kernels -------------------------------------------
//
// One thread per output row, mirroring matmul_q8_0. Block layouts match
// `src/kernels/simd.zig::dequantBlockQ{4,5,6}_K` exactly — the kernels read
// the same byte stream the CPU dequant reads.
//
// Reuses MMParams { m, k, w_offset_bytes } from matmul_q8_0.

constant uint Q4K_BLOCK_BYTES = 144;
constant uint Q4K_BLOCK_ELEMS = 256;
constant uint Q5K_BLOCK_BYTES = 176;
constant uint Q5K_BLOCK_ELEMS = 256;
constant uint Q6K_BLOCK_BYTES = 210;
constant uint Q6K_BLOCK_ELEMS = 256;

inline ushort kq_load_u16(device const uchar *p) {
    return (ushort)p[0] | ((ushort)p[1] << 8);
}

inline float kq_f16_to_f32(ushort bits) {
    return (float)as_type<half>(bits);
}

// Q4_K / Q5_K share a 12-byte 8-pair (sub_scale, sub_min) packing.
inline uchar2 q4k_scale_min(uint s, device const uchar *scales) {
    uchar sc, mn;
    if (s < 4) {
        sc = scales[s] & 0x3F;
        mn = scales[s + 4] & 0x3F;
    } else {
        sc = (scales[s + 4] & 0x0F) | ((scales[s - 4] >> 6) << 4);
        mn = (scales[s + 4] >> 4)   | ((scales[s] >> 6) << 4);
    }
    return uchar2(sc, mn);
}

kernel void matmul_q4_k(
    device float       *out      [[buffer(0)]],
    device const uchar *weights  [[buffer(1)]],
    device const float *acts     [[buffer(2)]],
    constant MMParams  &p        [[buffer(3)]],
    uint                gid      [[thread_position_in_grid]])
{
    if (gid >= p.m) return;
    uint blocks_per_row = p.k / Q4K_BLOCK_ELEMS;
    uint row_bytes = blocks_per_row * Q4K_BLOCK_BYTES;
    device const uchar *row = weights + p.w_offset_bytes + gid * row_bytes;

    float total = 0.0;
    for (uint b = 0; b < blocks_per_row; ++b) {
        device const uchar *blk = row + b * Q4K_BLOCK_BYTES;
        float d = kq_f16_to_f32(kq_load_u16(blk));
        float dmin = kq_f16_to_f32(kq_load_u16(blk + 2));
        device const uchar *scales = blk + 4;
        device const uchar *qs = blk + 16;
        device const float *x = acts + b * Q4K_BLOCK_ELEMS;

        // 4 pairs of sub-blocks: (0,1) (2,3) (4,5) (6,7). Pair `sp` shares
        // 32 qs bytes (lo nibble = sub-block sp*2, hi = sub-block sp*2+1).
        for (uint sp = 0; sp < 4; ++sp) {
            uchar2 sm0 = q4k_scale_min(sp * 2,     scales);
            uchar2 sm1 = q4k_scale_min(sp * 2 + 1, scales);
            float d1 = d    * (float)sm0.x;
            float m1 = dmin * (float)sm0.y;
            float d2 = d    * (float)sm1.x;
            float m2 = dmin * (float)sm1.y;
            uint y_idx = sp * 64;
            device const uchar *q = qs + sp * 32;
            for (uint l = 0; l < 32; ++l) {
                uchar packed = q[l];
                float deq_lo = d1 * (float)(packed & 0x0F) - m1;
                float deq_hi = d2 * (float)(packed >> 4)   - m2;
                total = fma(deq_lo, x[y_idx + l],      total);
                total = fma(deq_hi, x[y_idx + 32 + l], total);
            }
        }
    }
    out[gid] = total;
}

kernel void matmul_q5_k(
    device float       *out      [[buffer(0)]],
    device const uchar *weights  [[buffer(1)]],
    device const float *acts     [[buffer(2)]],
    constant MMParams  &p        [[buffer(3)]],
    uint                gid      [[thread_position_in_grid]])
{
    if (gid >= p.m) return;
    uint blocks_per_row = p.k / Q5K_BLOCK_ELEMS;
    uint row_bytes = blocks_per_row * Q5K_BLOCK_BYTES;
    device const uchar *row = weights + p.w_offset_bytes + gid * row_bytes;

    float total = 0.0;
    for (uint b = 0; b < blocks_per_row; ++b) {
        device const uchar *blk = row + b * Q5K_BLOCK_BYTES;
        float d = kq_f16_to_f32(kq_load_u16(blk));
        float dmin = kq_f16_to_f32(kq_load_u16(blk + 2));
        device const uchar *scales = blk + 4;
        device const uchar *qh = blk + 16;       // 32 high-bit bytes
        device const uchar *qs = blk + 48;       // 128 low-nibble bytes
        device const float *x = acts + b * Q5K_BLOCK_ELEMS;

        for (uint sp = 0; sp < 4; ++sp) {
            uint s0 = sp * 2;
            uint s1 = sp * 2 + 1;
            uchar2 sm0 = q4k_scale_min(s0, scales);
            uchar2 sm1 = q4k_scale_min(s1, scales);
            float d1 = d    * (float)sm0.x;
            float m1 = dmin * (float)sm0.y;
            float d2 = d    * (float)sm1.x;
            float m2 = dmin * (float)sm1.y;
            uchar mask_lo = (uchar)(1u << s0);
            uchar mask_hi = (uchar)(1u << s1);
            uint y_idx = sp * 64;
            device const uchar *q = qs + sp * 32;
            for (uint l = 0; l < 32; ++l) {
                uchar h = qh[l];
                uint w_lo = (uint)(q[l] & 0x0F) + (((h & mask_lo) != 0) ? 16u : 0u);
                uint w_hi = (uint)(q[l] >> 4)   + (((h & mask_hi) != 0) ? 16u : 0u);
                float deq_lo = d1 * (float)w_lo - m1;
                float deq_hi = d2 * (float)w_hi - m2;
                total = fma(deq_lo, x[y_idx + l],      total);
                total = fma(deq_hi, x[y_idx + 32 + l], total);
            }
        }
    }
    out[gid] = total;
}

kernel void matmul_q6_k(
    device float       *out      [[buffer(0)]],
    device const uchar *weights  [[buffer(1)]],
    device const float *acts     [[buffer(2)]],
    constant MMParams  &p        [[buffer(3)]],
    uint                gid      [[thread_position_in_grid]])
{
    if (gid >= p.m) return;
    uint blocks_per_row = p.k / Q6K_BLOCK_ELEMS;
    uint row_bytes = blocks_per_row * Q6K_BLOCK_BYTES;
    device const uchar *row = weights + p.w_offset_bytes + gid * row_bytes;

    float total = 0.0;
    for (uint b = 0; b < blocks_per_row; ++b) {
        device const uchar *blk = row + b * Q6K_BLOCK_BYTES;
        device const uchar *ql_all = blk;
        device const uchar *qh_all = blk + 128;
        device const char  *sc_all = (device const char *)(blk + 192);
        float d = kq_f16_to_f32(kq_load_u16(blk + 208));
        device const float *x = acts + b * Q6K_BLOCK_ELEMS;

        // Two halves of 128 weights each. Each l in [0,32) reconstructs four
        // weights at positions l, l+32, l+64, l+96 within the half.
        uint y_off = 0;
        uint ql_off = 0;
        uint qh_off = 0;
        uint sc_off = 0;
        for (uint hh = 0; hh < 2; ++hh) {
            for (uint l = 0; l < 32; ++l) {
                uint is = l / 16;
                uint ql_l   = ql_all[ql_off + l];
                uint ql_l32 = ql_all[ql_off + l + 32];
                uint qh_l   = qh_all[qh_off + l];
                int q1 = (int)((ql_l   & 0x0F) | (((qh_l >> 0) & 0x3) << 4)) - 32;
                int q2 = (int)((ql_l32 & 0x0F) | (((qh_l >> 2) & 0x3) << 4)) - 32;
                int q3 = (int)((ql_l   >> 4)   | (((qh_l >> 4) & 0x3) << 4)) - 32;
                int q4 = (int)((ql_l32 >> 4)   | (((qh_l >> 6) & 0x3) << 4)) - 32;
                float s0 = (float)sc_all[sc_off + is + 0];
                float s2 = (float)sc_all[sc_off + is + 2];
                float s4 = (float)sc_all[sc_off + is + 4];
                float s6 = (float)sc_all[sc_off + is + 6];
                total = fma(d * s0 * (float)q1, x[y_off + l + 0],  total);
                total = fma(d * s2 * (float)q2, x[y_off + l + 32], total);
                total = fma(d * s4 * (float)q3, x[y_off + l + 64], total);
                total = fma(d * s6 * (float)q4, x[y_off + l + 96], total);
            }
            y_off  += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
    out[gid] = total;
}

// Plain f32 copy. Caller binds dst+offset and src+offset; kernel just writes
// p.n contiguous floats from src to dst. Used to splice the just-computed
// k_cur / v_cur into the right slot in the persistent KV cache.
kernel void copy_f32(
    device float        *dst     [[buffer(0)]],
    device const float  *src     [[buffer(1)]],
    constant LenParams  &p       [[buffer(2)]],
    uint                 gid     [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    dst[gid] = src[gid];
}

// Attention scores: scores[h, t] = (q[h] · K[t, kv_h]) * inv_sqrt_hd.
// Grid: (n_heads, seq_len). One thread per (h, t).

struct AttnParams {
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint seq_len;
    float inv_sqrt_hd;
};

kernel void attn_scores(
    device float        *scores  [[buffer(0)]],
    device const float  *q       [[buffer(1)]],
    device const float  *k_cache [[buffer(2)]],
    constant AttnParams &p       [[buffer(3)]],
    uint2                gid     [[thread_position_in_grid]])
{
    if (gid.x >= p.n_heads || gid.y >= p.seq_len) return;
    uint h = gid.x;
    uint t = gid.y;
    uint kv_h = (h * p.n_kv_heads) / p.n_heads;
    uint kv_row = p.n_kv_heads * p.head_dim;

    device const float *qh = q + h * p.head_dim;
    device const float *kt = k_cache + t * kv_row + kv_h * p.head_dim;

    float s = 0.0;
    for (uint i = 0; i < p.head_dim; ++i) {
        s += qh[i] * kt[i];
    }
    scores[h * p.seq_len + t] = s * p.inv_sqrt_hd;
}

// Per-row numerically-stable softmax over `seq_len` columns. One threadgroup
// per row (head); threads cooperate via simdgroup reductions for max + sum.

kernel void softmax_rows(
    device float        *scores  [[buffer(0)]],
    constant AttnParams &p       [[buffer(1)]],
    uint                 hid     [[threadgroup_position_in_grid]],
    uint                 tid     [[thread_position_in_threadgroup]],
    uint                 ntg     [[threads_per_threadgroup]])
{
    device float *row = scores + hid * p.seq_len;
    threadgroup float scratch[32];
    uint sid = tid % 32;
    uint sgrp = tid / 32;
    uint n_sgrps = (ntg + 31) / 32;

    // Pass 1: max.
    float m = -INFINITY;
    for (uint i = tid; i < p.seq_len; i += ntg) {
        m = max(m, row[i]);
    }
    m = simd_max(m);
    if (sid == 0) scratch[sgrp] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgrp == 0) {
        m = (sid < n_sgrps) ? scratch[sid] : -INFINITY;
        m = simd_max(m);
        if (sid == 0) scratch[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = scratch[0];

    // Pass 2: write exp, accumulate sum.
    float s = 0.0;
    for (uint i = tid; i < p.seq_len; i += ntg) {
        float e = exp(row[i] - row_max);
        row[i] = e;
        s += e;
    }
    s = simd_sum(s);
    if (sid == 0) scratch[sgrp] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgrp == 0) {
        s = (sid < n_sgrps) ? scratch[sid] : 0.0;
        s = simd_sum(s);
        if (sid == 0) scratch[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = 1.0 / scratch[0];

    // Pass 3: normalize.
    for (uint i = tid; i < p.seq_len; i += ntg) {
        row[i] = row[i] * inv;
    }
}

// Weighted sum of V rows by softmaxed scores: out[h, d] = Σ_t scores[h,t]·V[t,kv_h,d].
// Grid: (n_heads, head_dim). One thread per (h, d).

kernel void attn_weighted_sum(
    device float        *out     [[buffer(0)]],
    device const float  *scores  [[buffer(1)]],
    device const float  *v_cache [[buffer(2)]],
    constant AttnParams &p       [[buffer(3)]],
    uint2                gid     [[thread_position_in_grid]])
{
    if (gid.x >= p.n_heads || gid.y >= p.head_dim) return;
    uint h = gid.x;
    uint d = gid.y;
    uint kv_h = (h * p.n_kv_heads) / p.n_heads;
    uint kv_row = p.n_kv_heads * p.head_dim;

    device const float *score_row = scores + h * p.seq_len;
    float acc = 0.0;
    for (uint t = 0; t < p.seq_len; ++t) {
        device const float *vt = v_cache + t * kv_row + kv_h * p.head_dim;
        acc += score_row[t] * vt[d];
    }
    out[h * p.head_dim + d] = acc;
}
)MSL";

// ----------------- Opaque handles --------------------------------------------

struct VtbMetalCtx {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> pso_q8_0;
    id<MTLComputePipelineState> pso_q4_k;
    id<MTLComputePipelineState> pso_q5_k;
    id<MTLComputePipelineState> pso_q6_k;
    id<MTLComputePipelineState> pso_rmsnorm;
    id<MTLComputePipelineState> pso_rope;
    id<MTLComputePipelineState> pso_swiglu;
    id<MTLComputePipelineState> pso_gelu_approx;
    id<MTLComputePipelineState> pso_residual;
    id<MTLComputePipelineState> pso_copy;
    id<MTLComputePipelineState> pso_attn_scores;
    id<MTLComputePipelineState> pso_softmax;
    id<MTLComputePipelineState> pso_attn_weighted;
    id<MTLComputePipelineState> pso_mlx_q4;
};

struct VtbMetalBuf {
    id<MTLBuffer> buffer;
};

struct VtbMetalSeg {
    VtbMetalCtx *ctx;
    id<MTLCommandBuffer> cmd;
    id<MTLComputeCommandEncoder> enc;
};

// ----------------- Lifecycle -------------------------------------------------

VtbMetalCtx *vtb_metal_init(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return NULL;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return NULL;

        NSError *err = nil;
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library = [device newLibraryWithSource:kMslSource options:opts error:&err];
        if (!library) {
            NSLog(@"vtb_metal_init: library compile failed: %@", err);
            return NULL;
        }

        // Compile every pipeline up front so subsequent dispatches are pure
        // kernel-launch overhead with no first-use latency.
        NSString *names[] = {
            @"matmul_q8_0", @"rmsnorm", @"rope_inplace",
            @"swiglu", @"residual_add", @"copy_f32",
            @"attn_scores", @"softmax_rows", @"attn_weighted_sum",
            @"matmul_mlx_q4",
            @"matmul_q4_k", @"matmul_q5_k", @"matmul_q6_k",
            @"gelu_approx",
        };
        const int n_psos = sizeof(names) / sizeof(names[0]);
        id<MTLComputePipelineState> psos[14] = {nil};
        for (int i = 0; i < n_psos; i++) {
            id<MTLFunction> fn = [library newFunctionWithName:names[i]];
            if (!fn) {
                NSLog(@"vtb_metal_init: function %@ not found", names[i]);
                return NULL;
            }
            psos[i] = [device newComputePipelineStateWithFunction:fn error:&err];
            if (!psos[i]) {
                NSLog(@"vtb_metal_init: pso %@ compile failed: %@", names[i], err);
                return NULL;
            }
        }

        VtbMetalCtx *ctx = (VtbMetalCtx *)calloc(1, sizeof(VtbMetalCtx));
        if (!ctx) return NULL;
        ctx->device = device;
        ctx->queue = queue;
        ctx->library = library;
        ctx->pso_q8_0 = psos[0];
        ctx->pso_rmsnorm = psos[1];
        ctx->pso_rope = psos[2];
        ctx->pso_swiglu = psos[3];
        ctx->pso_residual = psos[4];
        ctx->pso_copy = psos[5];
        ctx->pso_attn_scores = psos[6];
        ctx->pso_softmax = psos[7];
        ctx->pso_attn_weighted = psos[8];
        ctx->pso_mlx_q4 = psos[9];
        ctx->pso_q4_k = psos[10];
        ctx->pso_q5_k = psos[11];
        ctx->pso_q6_k = psos[12];
        ctx->pso_gelu_approx = psos[13];
        return ctx;
    }
}

void vtb_metal_destroy(VtbMetalCtx *ctx) {
    if (!ctx) return;
    ctx->device = nil;
    ctx->queue = nil;
    ctx->library = nil;
    ctx->pso_q8_0 = nil;
    ctx->pso_rmsnorm = nil;
    ctx->pso_rope = nil;
    ctx->pso_swiglu = nil;
    ctx->pso_residual = nil;
    ctx->pso_copy = nil;
    ctx->pso_attn_scores = nil;
    ctx->pso_softmax = nil;
    ctx->pso_attn_weighted = nil;
    ctx->pso_mlx_q4 = nil;
    ctx->pso_q4_k = nil;
    ctx->pso_q5_k = nil;
    ctx->pso_q6_k = nil;
    ctx->pso_gelu_approx = nil;
    free(ctx);
}

VtbMetalBuf *vtb_metal_alloc(VtbMetalCtx *ctx, size_t bytes, void **ptr_out) {
    if (!ctx || bytes == 0) return NULL;
    @autoreleasepool {
        id<MTLBuffer> b = [ctx->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!b) return NULL;
        VtbMetalBuf *out = (VtbMetalBuf *)calloc(1, sizeof(VtbMetalBuf));
        if (!out) return NULL;
        out->buffer = b;
        if (ptr_out) *ptr_out = [b contents];
        return out;
    }
}

VtbMetalBuf *vtb_metal_wrap(VtbMetalCtx *ctx, const void *ptr, size_t len) {
    if (!ctx || !ptr || len == 0) return NULL;
    @autoreleasepool {
        // Cast away const; we promise read-only on the GPU side. Deallocator
        // is nil because we don't own the underlying memory (mmap region is
        // owned by ModelMapper).
        id<MTLBuffer> b = [ctx->device newBufferWithBytesNoCopy:(void *)ptr
                                                        length:len
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];
        if (!b) return NULL;
        VtbMetalBuf *out = (VtbMetalBuf *)calloc(1, sizeof(VtbMetalBuf));
        if (!out) return NULL;
        out->buffer = b;
        return out;
    }
}

void vtb_metal_release(VtbMetalBuf *buf) {
    if (!buf) return;
    buf->buffer = nil;
    free(buf);
}

// ----------------- Q8_0 matmul -----------------------------------------------

int vtb_metal_matmul_q8_0(
    VtbMetalCtx *ctx,
    VtbMetalBuf *out_buf,
    VtbMetalBuf *w_buf,
    size_t w_offset,
    VtbMetalBuf *acts_buf,
    size_t m,
    size_t k)
{
    if (!ctx || !out_buf || !w_buf || !acts_buf) return 1;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx->pso_q8_0];
        [enc setBuffer:out_buf->buffer offset:0 atIndex:0];
        [enc setBuffer:w_buf->buffer offset:0 atIndex:1];
        [enc setBuffer:acts_buf->buffer offset:0 atIndex:2];

        struct MMParams { uint32_t m; uint32_t k; uint32_t w_offset_bytes; } params;
        params.m = (uint32_t)m;
        params.k = (uint32_t)k;
        params.w_offset_bytes = (uint32_t)w_offset;
        [enc setBytes:&params length:sizeof(params) atIndex:3];

        // One thread per output row. 64 threads per threadgroup.
        NSUInteger tg = 64;
        if (tg > ctx->pso_q8_0.maxTotalThreadsPerThreadgroup)
            tg = ctx->pso_q8_0.maxTotalThreadsPerThreadgroup;
        [enc dispatchThreads:MTLSizeMake(m, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.status == MTLCommandBufferStatusError) {
            NSLog(@"vtb_metal_matmul_q8_0: cmd failed: %@", cmd.error);
            return 2;
        }
        return 0;
    }
}

// ----------------- K-quant matmul dispatch -----------------------------------
//
// Q4_K / Q5_K / Q6_K kernels share the MMParams { m, k, w_offset_bytes }
// binding shape with Q8_0, so the dispatch pattern is identical — only the
// pipeline state differs. Each public entry point selects the PSO from the
// shared `kq_dispatch_*` helpers below.

static int kq_dispatch_oneshot(
    VtbMetalCtx *ctx,
    id<MTLComputePipelineState> pso,
    VtbMetalBuf *out_buf,
    VtbMetalBuf *w_buf,
    size_t w_offset,
    VtbMetalBuf *acts_buf,
    size_t m,
    size_t k,
    const char *label)
{
    if (!ctx || !out_buf || !w_buf || !acts_buf) return 1;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:out_buf->buffer offset:0 atIndex:0];
        [enc setBuffer:w_buf->buffer offset:0 atIndex:1];
        [enc setBuffer:acts_buf->buffer offset:0 atIndex:2];

        struct MMParams { uint32_t m; uint32_t k; uint32_t w_offset_bytes; } params;
        params.m = (uint32_t)m;
        params.k = (uint32_t)k;
        params.w_offset_bytes = (uint32_t)w_offset;
        [enc setBytes:&params length:sizeof(params) atIndex:3];

        NSUInteger tg = 64;
        if (tg > pso.maxTotalThreadsPerThreadgroup)
            tg = pso.maxTotalThreadsPerThreadgroup;
        [enc dispatchThreads:MTLSizeMake(m, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.status == MTLCommandBufferStatusError) {
            NSLog(@"%s: cmd failed: %@", label, cmd.error);
            return 2;
        }
        return 0;
    }
}

static void kq_dispatch_segment(
    VtbMetalSeg *seg,
    id<MTLComputePipelineState> pso,
    VtbMetalBuf *out_buf,
    VtbMetalBuf *w_buf,
    size_t w_offset,
    VtbMetalBuf *acts_buf,
    size_t m,
    size_t k)
{
    if (!seg) return;
    [seg->enc setComputePipelineState:pso];
    [seg->enc setBuffer:out_buf->buffer offset:0 atIndex:0];
    [seg->enc setBuffer:w_buf->buffer offset:0 atIndex:1];
    [seg->enc setBuffer:acts_buf->buffer offset:0 atIndex:2];
    struct MMParams { uint32_t m; uint32_t k; uint32_t w_offset_bytes; } params;
    params.m = (uint32_t)m;
    params.k = (uint32_t)k;
    params.w_offset_bytes = (uint32_t)w_offset;
    [seg->enc setBytes:&params length:sizeof(params) atIndex:3];
    NSUInteger tg = 64;
    if (tg > pso.maxTotalThreadsPerThreadgroup) tg = pso.maxTotalThreadsPerThreadgroup;
    [seg->enc dispatchThreads:MTLSizeMake(m, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

int vtb_metal_matmul_q4_k(VtbMetalCtx *ctx, VtbMetalBuf *out, VtbMetalBuf *w, size_t off, VtbMetalBuf *acts, size_t m, size_t k) {
    return kq_dispatch_oneshot(ctx, ctx ? ctx->pso_q4_k : nil, out, w, off, acts, m, k, "vtb_metal_matmul_q4_k");
}
int vtb_metal_matmul_q5_k(VtbMetalCtx *ctx, VtbMetalBuf *out, VtbMetalBuf *w, size_t off, VtbMetalBuf *acts, size_t m, size_t k) {
    return kq_dispatch_oneshot(ctx, ctx ? ctx->pso_q5_k : nil, out, w, off, acts, m, k, "vtb_metal_matmul_q5_k");
}
int vtb_metal_matmul_q6_k(VtbMetalCtx *ctx, VtbMetalBuf *out, VtbMetalBuf *w, size_t off, VtbMetalBuf *acts, size_t m, size_t k) {
    return kq_dispatch_oneshot(ctx, ctx ? ctx->pso_q6_k : nil, out, w, off, acts, m, k, "vtb_metal_matmul_q6_k");
}
void vtb_metal_segment_matmul_q4_k(VtbMetalSeg *seg, VtbMetalBuf *out, VtbMetalBuf *w, size_t off, VtbMetalBuf *acts, size_t m, size_t k) {
    kq_dispatch_segment(seg, seg ? seg->ctx->pso_q4_k : nil, out, w, off, acts, m, k);
}
void vtb_metal_segment_matmul_q5_k(VtbMetalSeg *seg, VtbMetalBuf *out, VtbMetalBuf *w, size_t off, VtbMetalBuf *acts, size_t m, size_t k) {
    kq_dispatch_segment(seg, seg ? seg->ctx->pso_q5_k : nil, out, w, off, acts, m, k);
}
void vtb_metal_segment_matmul_q6_k(VtbMetalSeg *seg, VtbMetalBuf *out, VtbMetalBuf *w, size_t off, VtbMetalBuf *acts, size_t m, size_t k) {
    kq_dispatch_segment(seg, seg ? seg->ctx->pso_q6_k : nil, out, w, off, acts, m, k);
}

// ----------------- Segment (batched) dispatch --------------------------------

VtbMetalSeg *vtb_metal_segment_begin(VtbMetalCtx *ctx) {
    if (!ctx) return NULL;
    @autoreleasepool {
        VtbMetalSeg *seg = (VtbMetalSeg *)calloc(1, sizeof(VtbMetalSeg));
        if (!seg) return NULL;
        seg->ctx = ctx;
        seg->cmd = [ctx->queue commandBuffer];
        seg->enc = [seg->cmd computeCommandEncoder];
        return seg;
    }
}

int vtb_metal_segment_commit(VtbMetalSeg *seg) {
    if (!seg) return 1;
    @autoreleasepool {
        [seg->enc endEncoding];
        [seg->cmd commit];
        // No waitUntilCompleted. The serial command queue chains this segment
        // after every prior submission, so subsequent segments observe its
        // writes without explicit fences. CPU reads of shared-storage buffers
        // must call vtb_metal_wait_idle first.
        seg->enc = nil;
        seg->cmd = nil;
        free(seg);
        return 0;
    }
}

void vtb_metal_wait_idle(VtbMetalCtx *ctx) {
    if (!ctx) return;
    @autoreleasepool {
        id<MTLCommandBuffer> drain = [ctx->queue commandBuffer];
        [drain commit];
        [drain waitUntilCompleted];
    }
}

void vtb_metal_segment_matmul_q8_0(
    VtbMetalSeg *seg,
    VtbMetalBuf *out_buf,
    VtbMetalBuf *w_buf,
    size_t w_offset,
    VtbMetalBuf *acts_buf,
    size_t m,
    size_t k)
{
    if (!seg) return;
    [seg->enc setComputePipelineState:seg->ctx->pso_q8_0];
    [seg->enc setBuffer:out_buf->buffer offset:0 atIndex:0];
    [seg->enc setBuffer:w_buf->buffer offset:0 atIndex:1];
    [seg->enc setBuffer:acts_buf->buffer offset:0 atIndex:2];
    struct MMParams { uint32_t m; uint32_t k; uint32_t w_offset_bytes; } params;
    params.m = (uint32_t)m;
    params.k = (uint32_t)k;
    params.w_offset_bytes = (uint32_t)w_offset;
    [seg->enc setBytes:&params length:sizeof(params) atIndex:3];
    NSUInteger tg = 64;
    if (tg > seg->ctx->pso_q8_0.maxTotalThreadsPerThreadgroup)
        tg = seg->ctx->pso_q8_0.maxTotalThreadsPerThreadgroup;
    [seg->enc dispatchThreads:MTLSizeMake(m, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void vtb_metal_segment_rmsnorm(
    VtbMetalSeg *seg,
    VtbMetalBuf *out_buf,
    VtbMetalBuf *in_buf,
    VtbMetalBuf *weight_buf,
    size_t weight_offset,
    size_t n,
    float eps)
{
    if (!seg) return;
    [seg->enc setComputePipelineState:seg->ctx->pso_rmsnorm];
    [seg->enc setBuffer:out_buf->buffer offset:0 atIndex:0];
    [seg->enc setBuffer:in_buf->buffer offset:0 atIndex:1];
    [seg->enc setBuffer:weight_buf->buffer offset:weight_offset atIndex:2];
    struct RmsParams { uint32_t n; float eps; } p = { (uint32_t)n, eps };
    [seg->enc setBytes:&p length:sizeof(p) atIndex:3];
    NSUInteger tg = 256;
    if (tg > seg->ctx->pso_rmsnorm.maxTotalThreadsPerThreadgroup)
        tg = seg->ctx->pso_rmsnorm.maxTotalThreadsPerThreadgroup;
    [seg->enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void vtb_metal_segment_rope(
    VtbMetalSeg *seg,
    VtbMetalBuf *x_buf,
    size_t n_heads,
    size_t head_dim,
    size_t pos,
    float base)
{
    if (!seg) return;
    [seg->enc setComputePipelineState:seg->ctx->pso_rope];
    [seg->enc setBuffer:x_buf->buffer offset:0 atIndex:0];
    struct RopeParams { uint32_t n_heads; uint32_t head_dim; uint32_t pos; float base; } p = {
        (uint32_t)n_heads, (uint32_t)head_dim, (uint32_t)pos, base,
    };
    [seg->enc setBytes:&p length:sizeof(p) atIndex:1];
    NSUInteger total = n_heads * (head_dim / 2);
    NSUInteger tg = 64;
    if (tg > seg->ctx->pso_rope.maxTotalThreadsPerThreadgroup)
        tg = seg->ctx->pso_rope.maxTotalThreadsPerThreadgroup;
    [seg->enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void vtb_metal_segment_swiglu(
    VtbMetalSeg *seg,
    VtbMetalBuf *gate_buf,
    VtbMetalBuf *up_buf,
    size_t n)
{
    if (!seg) return;
    [seg->enc setComputePipelineState:seg->ctx->pso_swiglu];
    [seg->enc setBuffer:gate_buf->buffer offset:0 atIndex:0];
    [seg->enc setBuffer:up_buf->buffer offset:0 atIndex:1];
    struct LenParams { uint32_t n; } p = { (uint32_t)n };
    [seg->enc setBytes:&p length:sizeof(p) atIndex:2];
    NSUInteger tg = 64;
    if (tg > seg->ctx->pso_swiglu.maxTotalThreadsPerThreadgroup)
        tg = seg->ctx->pso_swiglu.maxTotalThreadsPerThreadgroup;
    [seg->enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void vtb_metal_segment_gelu_approx(
    VtbMetalSeg *seg,
    VtbMetalBuf *gate_buf,
    VtbMetalBuf *up_buf,
    size_t n)
{
    if (!seg) return;
    [seg->enc setComputePipelineState:seg->ctx->pso_gelu_approx];
    [seg->enc setBuffer:gate_buf->buffer offset:0 atIndex:0];
    [seg->enc setBuffer:up_buf->buffer offset:0 atIndex:1];
    struct LenParams { uint32_t n; } p = { (uint32_t)n };
    [seg->enc setBytes:&p length:sizeof(p) atIndex:2];
    NSUInteger tg = 64;
    if (tg > seg->ctx->pso_gelu_approx.maxTotalThreadsPerThreadgroup)
        tg = seg->ctx->pso_gelu_approx.maxTotalThreadsPerThreadgroup;
    [seg->enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void vtb_metal_segment_residual_add(
    VtbMetalSeg *seg,
    VtbMetalBuf *a_buf,
    VtbMetalBuf *b_buf,
    size_t n)
{
    if (!seg) return;
    [seg->enc setComputePipelineState:seg->ctx->pso_residual];
    [seg->enc setBuffer:a_buf->buffer offset:0 atIndex:0];
    [seg->enc setBuffer:b_buf->buffer offset:0 atIndex:1];
    struct LenParams { uint32_t n; } p = { (uint32_t)n };
    [seg->enc setBytes:&p length:sizeof(p) atIndex:2];
    NSUInteger tg = 64;
    if (tg > seg->ctx->pso_residual.maxTotalThreadsPerThreadgroup)
        tg = seg->ctx->pso_residual.maxTotalThreadsPerThreadgroup;
    [seg->enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void vtb_metal_segment_copy(
    VtbMetalSeg *seg,
    VtbMetalBuf *dst_buf, size_t dst_offset_bytes,
    VtbMetalBuf *src_buf, size_t src_offset_bytes,
    size_t n_floats)
{
    if (!seg) return;
    [seg->enc setComputePipelineState:seg->ctx->pso_copy];
    [seg->enc setBuffer:dst_buf->buffer offset:dst_offset_bytes atIndex:0];
    [seg->enc setBuffer:src_buf->buffer offset:src_offset_bytes atIndex:1];
    struct LenParams { uint32_t n; } p = { (uint32_t)n_floats };
    [seg->enc setBytes:&p length:sizeof(p) atIndex:2];
    NSUInteger tg = 64;
    if (tg > seg->ctx->pso_copy.maxTotalThreadsPerThreadgroup)
        tg = seg->ctx->pso_copy.maxTotalThreadsPerThreadgroup;
    [seg->enc dispatchThreads:MTLSizeMake(n_floats, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

// Bind helper for the three attention kernels: builds the param block with
// the layer-pinned k/v cache offsets pre-applied via setBuffer:offset:.
struct AttnParams {
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t seq_len;
    float    inv_sqrt_hd;
};

void vtb_metal_segment_attn_scores(
    VtbMetalSeg *seg,
    VtbMetalBuf *scores_buf,
    VtbMetalBuf *q_buf,
    VtbMetalBuf *k_cache_buf, size_t k_cache_offset_bytes,
    size_t n_heads, size_t n_kv_heads, size_t head_dim, size_t seq_len,
    float inv_sqrt_hd)
{
    if (!seg) return;
    [seg->enc setComputePipelineState:seg->ctx->pso_attn_scores];
    [seg->enc setBuffer:scores_buf->buffer offset:0 atIndex:0];
    [seg->enc setBuffer:q_buf->buffer offset:0 atIndex:1];
    [seg->enc setBuffer:k_cache_buf->buffer offset:k_cache_offset_bytes atIndex:2];
    struct AttnParams p = {
        (uint32_t)n_heads, (uint32_t)n_kv_heads, (uint32_t)head_dim,
        (uint32_t)seq_len, inv_sqrt_hd,
    };
    [seg->enc setBytes:&p length:sizeof(p) atIndex:3];
    // 2D grid: (n_heads, seq_len). Threadgroup small — work per thread is
    // O(head_dim) which is tiny.
    NSUInteger tg_x = (n_heads <= 8) ? n_heads : 8;
    NSUInteger tg_y = 8;
    while (tg_x * tg_y > seg->ctx->pso_attn_scores.maxTotalThreadsPerThreadgroup) tg_y /= 2;
    [seg->enc dispatchThreads:MTLSizeMake(n_heads, seq_len, 1)
              threadsPerThreadgroup:MTLSizeMake(tg_x, tg_y, 1)];
}

void vtb_metal_segment_softmax_rows(
    VtbMetalSeg *seg,
    VtbMetalBuf *scores_buf,
    size_t n_heads, size_t n_kv_heads, size_t head_dim, size_t seq_len)
{
    if (!seg) return;
    (void)head_dim; (void)n_kv_heads; // params bag shared with attn_scores
    [seg->enc setComputePipelineState:seg->ctx->pso_softmax];
    [seg->enc setBuffer:scores_buf->buffer offset:0 atIndex:0];
    struct AttnParams p = {
        (uint32_t)n_heads, (uint32_t)n_kv_heads, (uint32_t)head_dim,
        (uint32_t)seq_len, 0.0f,
    };
    [seg->enc setBytes:&p length:sizeof(p) atIndex:1];
    // One threadgroup per head row; 256 threads cooperate on the reduction.
    NSUInteger tg = 256;
    if (tg > seg->ctx->pso_softmax.maxTotalThreadsPerThreadgroup)
        tg = seg->ctx->pso_softmax.maxTotalThreadsPerThreadgroup;
    [seg->enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void vtb_metal_segment_attn_weighted_sum(
    VtbMetalSeg *seg,
    VtbMetalBuf *out_buf,
    VtbMetalBuf *scores_buf,
    VtbMetalBuf *v_cache_buf, size_t v_cache_offset_bytes,
    size_t n_heads, size_t n_kv_heads, size_t head_dim, size_t seq_len)
{
    if (!seg) return;
    [seg->enc setComputePipelineState:seg->ctx->pso_attn_weighted];
    [seg->enc setBuffer:out_buf->buffer offset:0 atIndex:0];
    [seg->enc setBuffer:scores_buf->buffer offset:0 atIndex:1];
    [seg->enc setBuffer:v_cache_buf->buffer offset:v_cache_offset_bytes atIndex:2];
    struct AttnParams p = {
        (uint32_t)n_heads, (uint32_t)n_kv_heads, (uint32_t)head_dim,
        (uint32_t)seq_len, 0.0f,
    };
    [seg->enc setBytes:&p length:sizeof(p) atIndex:3];
    // 2D grid: (n_heads, head_dim). Per-thread loop over seq_len.
    NSUInteger tg_x = (n_heads <= 8) ? n_heads : 8;
    NSUInteger tg_y = 8;
    while (tg_x * tg_y > seg->ctx->pso_attn_weighted.maxTotalThreadsPerThreadgroup) tg_y /= 2;
    [seg->enc dispatchThreads:MTLSizeMake(n_heads, head_dim, 1)
              threadsPerThreadgroup:MTLSizeMake(tg_x, tg_y, 1)];
}

void vtb_metal_segment_matmul_mlx_q4(
    VtbMetalSeg *seg,
    VtbMetalBuf *out_buf,
    VtbMetalBuf *weight_buf, size_t weight_offset,
    VtbMetalBuf *scales_buf, size_t scales_offset,
    VtbMetalBuf *biases_buf, size_t biases_offset,
    VtbMetalBuf *acts_buf,   size_t acts_offset,
    size_t m, size_t k,
    uint32_t group_size,
    uint32_t scale_dtype_is_bf16)
{
    if (!seg) return;
    // Snap weight/scales/biases offsets to 4-byte alignment; pass remainder
    // bytes via the params struct. Mirrors `vtb_metal_matmul_mlx_q4` — see
    // that function for the safetensors-misalignment rationale.
    const size_t w_aligned  = weight_offset & ~(size_t)3;
    const size_t s_aligned  = scales_offset & ~(size_t)3;
    const size_t b_aligned  = biases_offset & ~(size_t)3;
    const uint32_t w_resid  = (uint32_t)(weight_offset - w_aligned);
    const uint32_t s_resid  = (uint32_t)(scales_offset - s_aligned);
    const uint32_t b_resid  = (uint32_t)(biases_offset - b_aligned);

    [seg->enc setComputePipelineState:seg->ctx->pso_mlx_q4];
    [seg->enc setBuffer:out_buf->buffer    offset:0           atIndex:0];
    [seg->enc setBuffer:weight_buf->buffer offset:w_aligned   atIndex:1];
    [seg->enc setBuffer:scales_buf->buffer offset:s_aligned   atIndex:2];
    [seg->enc setBuffer:biases_buf->buffer offset:b_aligned   atIndex:3];
    // acts buffer is f32 scratch — always 4-byte aligned, no snap needed.
    [seg->enc setBuffer:acts_buf->buffer   offset:acts_offset atIndex:4];

    struct MlxQ4ParamsHost p = {
        .m = (uint32_t)m,
        .k = (uint32_t)k,
        .w_offset_bytes = w_resid,
        .scales_offset_bytes = s_resid,
        .biases_offset_bytes = b_resid,
        .group_size = group_size,
        .scale_dtype_is_bf16 = scale_dtype_is_bf16,
    };
    [seg->enc setBytes:&p length:sizeof(p) atIndex:5];

    NSUInteger tg = 64;
    if (tg > seg->ctx->pso_mlx_q4.maxTotalThreadsPerThreadgroup)
        tg = seg->ctx->pso_mlx_q4.maxTotalThreadsPerThreadgroup;
    [seg->enc dispatchThreads:MTLSizeMake(m, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}
