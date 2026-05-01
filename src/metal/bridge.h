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

#ifdef __cplusplus
}
#endif

#endif // VTB_METAL_BRIDGE_H
