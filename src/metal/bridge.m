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
        // Read f16 scale (bytes 0..2, little-endian).
        ushort scale_u16 = (ushort)blk[0] | ((ushort)blk[1] << 8);
        half scale = as_type<half>(scale_u16);
        float scale_f = (float)scale;
        // Quants are i8 packed at bytes 2..34.
        device const char *qs = (device const char*)(blk + 2);
        device const float *x = acts + b * Q8_BLOCK_ELEMS;

        float block_dot = 0.0;
        for (uint j = 0; j < Q8_BLOCK_ELEMS; ++j) {
            block_dot += (float)qs[j] * x[j];
        }
        total += scale_f * block_dot;
    }
    out[gid] = total;
}
)MSL";

// ----------------- Opaque handles --------------------------------------------

struct VtbMetalCtx {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> pso_q8_0;
};

struct VtbMetalBuf {
    id<MTLBuffer> buffer;
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

        id<MTLFunction> fn_q8_0 = [library newFunctionWithName:@"matmul_q8_0"];
        if (!fn_q8_0) {
            NSLog(@"vtb_metal_init: matmul_q8_0 function not found");
            return NULL;
        }
        id<MTLComputePipelineState> pso_q8_0 = [device newComputePipelineStateWithFunction:fn_q8_0 error:&err];
        if (!pso_q8_0) {
            NSLog(@"vtb_metal_init: pso compile failed: %@", err);
            return NULL;
        }

        VtbMetalCtx *ctx = (VtbMetalCtx *)calloc(1, sizeof(VtbMetalCtx));
        if (!ctx) return NULL;
        ctx->device = device;
        ctx->queue = queue;
        ctx->library = library;
        ctx->pso_q8_0 = pso_q8_0;
        return ctx;
    }
}

void vtb_metal_destroy(VtbMetalCtx *ctx) {
    if (!ctx) return;
    ctx->device = nil;
    ctx->queue = nil;
    ctx->library = nil;
    ctx->pso_q8_0 = nil;
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

        // One thread per output row. Threadgroup of 64 is a reasonable
        // default; tune later.
        NSUInteger tg = 64;
        if (tg > ctx->pso_q8_0.maxTotalThreadsPerThreadgroup) {
            tg = ctx->pso_q8_0.maxTotalThreadsPerThreadgroup;
        }
        MTLSize grid = MTLSizeMake(m, 1, 1);
        MTLSize threads = MTLSizeMake(tg, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:threads];
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
