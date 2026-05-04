# MLX-Q4 GPU Dispatch Implementation Plan (Plan C)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the already-compiled `matmul_mlx_q4` Metal kernel into the runtime forward path so MLX-format Llama models (e.g. `mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit`) run their Q4 matmuls on the Apple GPU instead of the CPU baseline in `kernels/mlx.zig`.

**Architecture:** Refactor `MetalBackend` to hold a per-shard array of `MTLBuffer` wrappers around each mmap'd safetensors shard (instead of one buffer wrapping a single GGUF mmap). Add a pointer→(buffer, offset) lookup used by `forward.zig::matmulRuntime` to translate weight/scales/biases slice pointers into Metal buffer handles + byte offsets. Expose the existing inline `matmul_mlx_q4` MSL kernel via a new C-ABI function `vtb_metal_segment_matmul_mlx_q4` (4-buffer dispatch: out, weight, scales, biases + acts), wrap it in Zig, and route `mlx_q4` quant types through it when an `HfBundle`-backed `MetalBackend` is present. The GGUF (single-shard) path stays bit-identical; full-GPU forward (rmsnorm/RoPE/attn) for MLX models is **out of scope** for this plan — only matmul moves to GPU.

**Tech Stack:** Zig 0.16+, Objective-C bridge (`src/metal/bridge.m`) + MSL kernels, Apple Metal API, safetensors v1, mmap.

---

## Pre-flight

Run from repo root. Verify clean baseline before starting.

```sh
cd /Users/chrisolson/development/github/vantablack
git status                      # must be clean
zig version                     # must be 0.16.0+
zig build test                  # current tests must pass
zig build -Dmetal=true          # current Metal build must compile
```

If any step fails, stop and resolve before starting Task 1.

**Test asset:** Tasks 8–9 require a local MLX TinyLlama snapshot. If absent:

```sh
huggingface-cli download mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit \
  --local-dir ~/models/tinyllama-mlx-4bit
```

The snapshot path is referenced by `MLX_TEST_DIR` env var in tests.

---

## File Structure

| File | Role | Change |
|------|------|--------|
| `src/runtime/metal_backend.zig` | Persistent Metal state, weight buffer wrapping | Refactor: replace single `weights_buf` with `weight_shards: []WeightShard`; add `initFromHf`; add `resolveWeight(ptr) -> ?WeightLoc` |
| `src/metal/bridge.h` | Flat C ABI | Add `vtb_metal_segment_matmul_mlx_q4`, `vtb_metal_matmul_mlx_q4` decls |
| `src/metal/bridge.m` | Obj-C + MSL bridge | Add host dispatch funcs that bind out/weight/scales/biases/acts buffers + `MlxQ4Params` and encode `pso_mlx_q4` |
| `src/metal/bridge.zig` | Zig FFI shim | Add `extern` decls + `Device.matmulMlxQ4`, `Segment.matmulMlxQ4` methods |
| `src/runtime/forward.zig` | Per-token forward | In `matmulRuntime`, route `.mlx_q4` to GPU when `MetalBackend` present and acts/out alias scratch |
| `src/main.zig` | CLI | In `runHfCli`, construct `MetalBackend.initFromHf` when `metal_enabled`, pass into `State.init` and forward `step` calls |
| `src/runtime/server.zig` | HTTP server | `Server.initFromHf` accepts optional `*MetalBackend`, threaded through |
| `tests/mlx_gpu_test.zig` *(new)* | Integration smoke test | Builds an HfBundle-like fixture, dispatches GPU matmul, compares to CPU baseline within ULP tolerance |

---

## Task 1: Refactor MetalBackend to hold per-shard weight buffers

**Files:**
- Modify: `src/runtime/metal_backend.zig`

The current backend wraps one mmap as `weights_buf` and resolves any weight slice via `weightOffset(bytes) = bytes.ptr - weights_base`. Multi-shard breaks that single-base assumption. Replace with an array.

- [ ] **Step 1: Write a failing test that constructs MetalBackend with two synthetic shards**

Add to bottom of `src/runtime/metal_backend.zig`:

```zig
test "MetalBackend.resolveWeight: two shards, pointer routes to correct buffer" {
    if (!metal.metal_enabled) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    // Two page-aligned dummy shards.
    const page = std.heap.pageSize();
    const shard0 = try std.posix.mmap(null, page, std.posix.PROT.READ | std.posix.PROT.WRITE,
        .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(shard0);
    const shard1 = try std.posix.mmap(null, page, std.posix.PROT.READ | std.posix.PROT.WRITE,
        .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(shard1);
    shard0[0] = 0xAA;
    shard1[0] = 0xBB;

    const shards = [_]ShardInput{
        .{ .bytes = shard0 },
        .{ .bytes = shard1 },
    };

    var be = try MetalBackend.initShards(allocator, &shards, fakeCfg());
    defer be.deinit(allocator);

    const loc0 = be.resolveWeight(shard0.ptr).?;
    const loc1 = be.resolveWeight(shard1.ptr + 64).?;
    try std.testing.expectEqual(@as(usize, 0), loc0.offset);
    try std.testing.expectEqual(@as(usize, 64), loc1.offset);
    try std.testing.expect(loc0.buf != loc1.buf);

    try std.testing.expect(be.resolveWeight(@as([*]const u8, @ptrFromInt(0xdead0000))) == null);
}
```

(Define a `fakeCfg()` helper that returns a `model_mod.LlamaConfig` with minimal valid dims — see Step 4.)

- [ ] **Step 2: Run the test to confirm failure**

```sh
zig build test -Dmetal=true 2>&1 | grep -E "(FAIL|error|MetalBackend.resolveWeight)"
```

Expected: compilation error referencing `ShardInput`, `initShards`, or `resolveWeight` — these don't exist yet.

- [ ] **Step 3: Add the new types and methods**

Replace the `MetalBackend` struct fields (and `init`) with:

```zig
pub const ShardInput = struct {
    bytes: []const u8,    // page-aligned mmap region
};

pub const WeightShard = struct {
    buf: *metal.Buf,
    base: [*]const u8,
    len: usize,
};

pub const WeightLoc = struct {
    buf: *metal.Buf,
    offset: usize,
};

pub const MetalBackend = struct {
    dev: metal.Device,
    weight_shards: []WeightShard,    // owns: array allocated; bufs released in deinit
    // ... existing scratch/KV fields unchanged ...

    pub fn initShards(
        allocator: Allocator,
        shards: []const ShardInput,
        cfg: model_mod.LlamaConfig,
    ) InitError!MetalBackend {
        var dev = try metal.Device.init();
        errdefer dev.deinit();

        const ws = try allocator.alloc(WeightShard, shards.len);
        errdefer allocator.free(ws);

        var wrapped: usize = 0;
        errdefer for (ws[0..wrapped]) |*s| dev.release(s.buf);

        for (shards, 0..) |sh, i| {
            const len_aligned = std.mem.alignForward(usize, sh.bytes.len, std.heap.pageSize());
            if (len_aligned == 0) return InitError.OutOfMemory;
            const buf = dev.wrap(sh.bytes[0..len_aligned]) catch return InitError.OutOfMemory;
            ws[i] = .{ .buf = buf, .base = sh.bytes.ptr, .len = len_aligned };
            wrapped += 1;
        }

        // ... allocate scratch + KV exactly as before ...
        return .{
            .dev = dev,
            .weight_shards = ws,
            // ... rest of fields unchanged ...
        };
    }

    pub fn resolveWeight(self: *const MetalBackend, ptr: [*]const u8) ?WeightLoc {
        const addr = @intFromPtr(ptr);
        for (self.weight_shards) |sh| {
            const base_addr = @intFromPtr(sh.base);
            if (addr >= base_addr and addr < base_addr + sh.len) {
                return .{ .buf = sh.buf, .offset = addr - base_addr };
            }
        }
        return null;
    }

    // Adapt the existing weightOffset to keep GGUF callers working:
    pub fn weightOffset(self: *const MetalBackend, bytes: []const u8) usize {
        // GGUF path is shard 0.
        std.debug.assert(self.weight_shards.len >= 1);
        const base = @intFromPtr(self.weight_shards[0].base);
        return @intFromPtr(bytes.ptr) - base;
    }
};
```

In `deinit`, replace the single `dev.release(self.weights_buf)` with `for (self.weight_shards) |s| dev.release(s.buf); allocator.free(self.weight_shards);`.

Also expose `weights_buf` for legacy GGUF call sites: add a `pub fn weightsBuf(self: *const MetalBackend) *metal.Buf { return self.weight_shards[0].buf; }` and replace every `mb.weights_buf` reference with `mb.weightsBuf()` (forward.zig has 16 such call sites — see grep below).

- [ ] **Step 4: Adapt the existing `init(mapper, cfg)` to call `initShards`**

```zig
pub fn init(
    allocator: Allocator,
    mapper: *const mapper_mod.ModelMapper,
    cfg: model_mod.LlamaConfig,
) InitError!MetalBackend {
    const shards = [_]ShardInput{ .{ .bytes = mapper.mapped } };
    return initShards(allocator, &shards, cfg);
}
```

This keeps every GGUF caller (main.zig, server.zig) source-compatible.

- [ ] **Step 5: Update forward.zig call sites that use `mb.weights_buf`**

```sh
grep -n "mb\.weights_buf" src/runtime/forward.zig
```

Replace each `mb.weights_buf` with `mb.weightsBuf()`. Pattern is mechanical — same callsite, no semantic change.

- [ ] **Step 6: Run tests**

```sh
zig build test -Dmetal=true
```

Expected: all tests pass, including the new `resolveWeight` test. If GGUF tests break, the `weightsBuf()` shim is wrong — recheck.

- [ ] **Step 7: Commit**

```sh
git add src/runtime/metal_backend.zig src/runtime/forward.zig
git commit -m "refactor(metal): hold weight buffers as per-shard array

Replaces single weights_buf with weight_shards: []WeightShard plus a
resolveWeight(ptr) -> ?WeightLoc lookup. GGUF path becomes a one-shard
special case via init(mapper, cfg). No behavior change."
```

---

## Task 2: Construct MetalBackend from an HfBundle

**Files:**
- Modify: `src/runtime/metal_backend.zig`
- Modify: `src/core/hf_loader.zig` (read-only — already exposes `shards: []Shard`)

- [ ] **Step 1: Add a failing test that builds backend from a 2-shard HfBundle stub**

In `src/runtime/metal_backend.zig`:

```zig
test "MetalBackend.initFromHf: wraps every shard's mmap" {
    if (!metal.metal_enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var bundle = try fakeBundle(allocator, &.{
        // 2 fake shards, page-aligned dummy bytes
    });
    defer bundle.deinit();

    var be = try MetalBackend.initFromHf(allocator, &bundle, fakeCfg());
    defer be.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), be.weight_shards.len);
}
```

- [ ] **Step 2: Run to confirm failure**

```sh
zig build test -Dmetal=true 2>&1 | tail -20
```

Expected: `error: initFromHf is not a member of MetalBackend`.

- [ ] **Step 3: Implement `initFromHf`**

```zig
const hf_loader_mod = @import("../core/hf_loader.zig");

pub fn initFromHf(
    allocator: Allocator,
    bundle: *const hf_loader_mod.HfBundle,
    cfg: model_mod.LlamaConfig,
) InitError!MetalBackend {
    const ins = try allocator.alloc(ShardInput, bundle.shards.len);
    defer allocator.free(ins);
    for (bundle.shards, 0..) |sh, i| ins[i] = .{ .bytes = sh.mapped };
    return initShards(allocator, ins, cfg);
}
```

- [ ] **Step 4: Run tests**

```sh
zig build test -Dmetal=true
```

Expected: pass.

- [ ] **Step 5: Commit**

```sh
git add src/runtime/metal_backend.zig
git commit -m "feat(metal): MetalBackend.initFromHf wraps every HF shard"
```

---

## Task 3: Expose `matmul_mlx_q4` via C ABI

**Files:**
- Modify: `src/metal/bridge.h`
- Modify: `src/metal/bridge.m`

The MSL kernel `matmul_mlx_q4` and its `pso_mlx_q4` pipeline state already exist (bridge.m line 213, line 406 / line 443 in the registration array). Only the host-side dispatch wrapper is missing.

- [ ] **Step 1: Add the C declarations to `src/metal/bridge.h`**

After the existing `vtb_metal_segment_attn_weighted_sum` decl (~line 127), append:

```c
// MLX-format 4-bit affine block-quant matmul. Weight, scales, biases may
// live in different MTLBuffers (different safetensors shards).
//
//   out_buf:          shared-storage scratch, m floats
//   weight_buf,off:   packed U32 nibbles, m * (k/8) u32s (m * k/2 bytes)
//   scales_buf,off:   F16 or BF16 scales, m * (k/group_size) entries
//   biases_buf,off:   same layout as scales, F16 or BF16
//   acts_buf,off:     k floats
//   m, k:             dims
//   group_size:       quant group (typically 64)
//   scale_dtype:      0 = f16, 1 = bf16
void vtb_metal_segment_matmul_mlx_q4(
    void *seg,
    void *out_buf,
    void *weight_buf, size_t weight_off,
    void *scales_buf, size_t scales_off,
    void *biases_buf, size_t biases_off,
    void *acts_buf,   size_t acts_off,
    size_t m, size_t k,
    uint32_t group_size,
    uint32_t scale_dtype_is_bf16);

// Single-shot variant (no segment, ends its own command buffer).
int vtb_metal_matmul_mlx_q4(
    void *ctx,
    void *out_buf,
    void *weight_buf, size_t weight_off,
    void *scales_buf, size_t scales_off,
    void *biases_buf, size_t biases_off,
    void *acts_buf,   size_t acts_off,
    size_t m, size_t k,
    uint32_t group_size,
    uint32_t scale_dtype_is_bf16);
```

- [ ] **Step 2: Implement segment dispatch in `src/metal/bridge.m`**

Right after `vtb_metal_segment_matmul_q8_0` definition, add:

```objc
void vtb_metal_segment_matmul_mlx_q4(
    void *seg_v,
    void *out_buf, void *w_buf, size_t w_off,
    void *s_buf, size_t s_off, void *b_buf, size_t b_off,
    void *a_buf, size_t a_off,
    size_t m, size_t k,
    uint32_t group_size, uint32_t scale_is_bf16)
{
    VtbMetalSegment *seg = (__bridge VtbMetalSegment *)seg_v;
    id<MTLComputeCommandEncoder> enc = seg->enc;

    [enc setComputePipelineState:seg->ctx->pso_mlx_q4];
    [enc setBuffer:(__bridge id<MTLBuffer>)out_buf offset:0           atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w_buf   offset:w_off       atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)s_buf   offset:s_off       atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)b_buf   offset:b_off       atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)a_buf   offset:a_off       atIndex:4];

    struct MlxQ4Params p = {
        .m = (uint)m, .k = (uint)k,
        .w_offset_bytes = 0,    // baked into setBuffer offset
        .scales_offset_bytes = 0,
        .biases_offset_bytes = 0,
        .group_size = group_size,
        .scale_dtype_is_bf16 = scale_is_bf16,
    };
    [enc setBytes:&p length:sizeof(p) atIndex:5];

    NSUInteger tg = MIN((NSUInteger)64, seg->ctx->pso_mlx_q4.maxTotalThreadsPerThreadgroup);
    NSUInteger groups = (m + tg - 1) / tg;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}
```

- [ ] **Step 3: Implement the one-shot variant**

```objc
int vtb_metal_matmul_mlx_q4(
    void *ctx_v,
    void *out_buf, void *w_buf, size_t w_off,
    void *s_buf, size_t s_off, void *b_buf, size_t b_off,
    void *a_buf, size_t a_off,
    size_t m, size_t k,
    uint32_t group_size, uint32_t scale_is_bf16)
{
    VtbMetalCtx *ctx = (VtbMetalCtx *)ctx_v;
    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    [enc setComputePipelineState:ctx->pso_mlx_q4];
    [enc setBuffer:(__bridge id<MTLBuffer>)out_buf offset:0      atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)w_buf   offset:w_off  atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)s_buf   offset:s_off  atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)b_buf   offset:b_off  atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)a_buf   offset:a_off  atIndex:4];

    struct MlxQ4Params p = {
        .m = (uint)m, .k = (uint)k,
        .w_offset_bytes = 0, .scales_offset_bytes = 0, .biases_offset_bytes = 0,
        .group_size = group_size, .scale_dtype_is_bf16 = scale_is_bf16,
    };
    [enc setBytes:&p length:sizeof(p) atIndex:5];

    NSUInteger tg = MIN((NSUInteger)64, ctx->pso_mlx_q4.maxTotalThreadsPerThreadgroup);
    NSUInteger groups = (m + tg - 1) / tg;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return cb.status == MTLCommandBufferStatusCompleted ? 0 : -1;
}
```

**Why baked offsets:** Metal's `setBuffer:offset:` already shifts the buffer's effective base, so the kernel's internal `w_offset_bytes` field is redundant in this dispatch path. Keep the struct field for forward compatibility but set it to 0.

- [ ] **Step 4: Build with metal**

```sh
zig build -Dmetal=true 2>&1 | tail -10
```

Expected: clean build. If linker complains about missing symbols, the .h decls don't match the .m definitions exactly — recheck signatures.

- [ ] **Step 5: Commit**

```sh
git add src/metal/bridge.h src/metal/bridge.m
git commit -m "feat(metal): expose vtb_metal_(segment_)matmul_mlx_q4

Wraps the existing pso_mlx_q4 pipeline as a 4-buffer dispatch
(out, weight, scales, biases, acts) so callers can supply weight
shards from different MTLBuffers."
```

---

## Task 4: Add Zig wrappers in bridge.zig

**Files:**
- Modify: `src/metal/bridge.zig`

- [ ] **Step 1: Add a failing test**

In `src/metal/bridge.zig`:

```zig
test "Device.matmulMlxQ4 dispatches without error" {
    if (!metal_enabled) return error.SkipZigTest;
    var dev = try Device.init();
    defer dev.deinit();
    // 4x8 row-major Q4 matmul fixture
    const m: usize = 4;
    const k: usize = 64;
    var out_host: [m]f32 = undefined;
    const out = try dev.alloc(m * @sizeOf(f32));
    defer dev.release(out);
    // ... fill weight/scales/biases/acts buffers with deterministic values ...
    try dev.matmulMlxQ4(out, w_buf, 0, s_buf, 0, b_buf, 0, a_buf, 0, m, k, 64, 0);
    @memcpy(&out_host, @as([*]const f32, @ptrCast(@alignCast(dev.bufContents(out))))[0..m]);
    // Just smoke: outputs are finite.
    for (out_host) |v| try std.testing.expect(std.math.isFinite(v));
}
```

- [ ] **Step 2: Add extern decls + Device/Segment methods**

Add to extern block (near line 16):

```zig
extern fn vtb_metal_segment_matmul_mlx_q4(
    seg: *anyopaque,
    out_buf: *anyopaque,
    w_buf: *anyopaque, w_off: usize,
    s_buf: *anyopaque, s_off: usize,
    b_buf: *anyopaque, b_off: usize,
    a_buf: *anyopaque, a_off: usize,
    m: usize, k: usize,
    group_size: u32, scale_dt_is_bf16: u32,
) void;

extern fn vtb_metal_matmul_mlx_q4(
    ctx: *anyopaque,
    out_buf: *anyopaque,
    w_buf: *anyopaque, w_off: usize,
    s_buf: *anyopaque, s_off: usize,
    b_buf: *anyopaque, b_off: usize,
    a_buf: *anyopaque, a_off: usize,
    m: usize, k: usize,
    group_size: u32, scale_dt_is_bf16: u32,
) c_int;
```

Add a method on `Device`:

```zig
pub fn matmulMlxQ4(
    self: *Device,
    out: *Buf,
    w: *Buf, w_off: usize,
    s: *Buf, s_off: usize,
    b: *Buf, b_off: usize,
    a: *Buf, a_off: usize,
    m: usize, k: usize,
    group_size: u32,
    scale_is_bf16: bool,
) !void {
    const rc = vtb_metal_matmul_mlx_q4(
        self.ctx, out, w, w_off, s, s_off, b, b_off, a, a_off,
        m, k, group_size, if (scale_is_bf16) 1 else 0,
    );
    if (rc != 0) return error.MetalDispatchFailed;
}
```

And on `Segment`:

```zig
pub fn matmulMlxQ4(
    self: Segment,
    out: *Buf,
    w: *Buf, w_off: usize,
    s: *Buf, s_off: usize,
    b: *Buf, b_off: usize,
    a: *Buf, a_off: usize,
    m: usize, k: usize,
    group_size: u32,
    scale_is_bf16: bool,
) void {
    vtb_metal_segment_matmul_mlx_q4(
        self.handle, out, w, w_off, s, s_off, b, b_off, a, a_off,
        m, k, group_size, if (scale_is_bf16) 1 else 0,
    );
}
```

- [ ] **Step 3: Run tests**

```sh
zig build test -Dmetal=true
```

Expected: pass.

- [ ] **Step 4: Commit**

```sh
git add src/metal/bridge.zig
git commit -m "feat(metal): Zig wrappers for matmul_mlx_q4 (Device + Segment)"
```

---

## Task 5: Backend-level matmul helper for MLX-Q4

**Files:**
- Modify: `src/runtime/metal_backend.zig`

- [ ] **Step 1: Add a failing test**

Synthetic case: 2 shards. Weight in shard 0, scales/biases in shard 1. Activations in a scratch buffer.

```zig
test "MetalBackend.matmulMlxQ4: routes weight + scales + biases across shards" {
    if (!metal.metal_enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // m=4, k=64, group_size=64 -> 1 group/row.
    const m: usize = 4;
    const k: usize = 64;
    const w_bytes = m * k / 2;          // 4-bit packed
    const sb_bytes = m * 2;             // 1 group/row, 2 bytes each (f16)

    // Build fake shards (page-aligned mmap'd anon).
    // ... see helper fakeShards ...
    // Fill weight: every nibble = 0x1 → q=1.
    // Scales: 1.0 (f16 = 0x3C00).
    // Biases: 0.0.
    // Acts: 1.0 in slot 0, 0 elsewhere.
    //
    // Expected per-row output: scale * Σ(q*x - 8*x) + bias * Σ x
    //                        = 1 * (1*1 + 8*0... - 8*1) = -7

    // ... allocate scratch acts + out via dev.alloc ...
    try be.matmulMlxQ4(out_buf, weight_ptr, scales_ptr, biases_ptr, acts_buf, m, k, 4, 64, .f16);

    const out = @as([*]const f32, @ptrCast(@alignCast(be.dev.bufContents(out_buf))))[0..m];
    for (out) |v| try std.testing.expectApproxEqAbs(@as(f32, -7), v, 1e-4);
}
```

- [ ] **Step 2: Implement the helper**

Add to `MetalBackend`:

```zig
pub fn matmulMlxQ4(
    self: *MetalBackend,
    out: *metal.Buf,
    weight_ptr: [*]const u8,
    scales_ptr: [*]const u8,
    biases_ptr: [*]const u8,
    acts: *metal.Buf,
    m: usize, k: usize,
    bits: u32,
    group_size: u32,
    scale_dtype: enum { f16, bf16 },
) !void {
    // Plan C only handles 4-bit. Other widths stay on CPU.
    if (bits != 4) return error.UnsupportedBits;

    const w = self.resolveWeight(weight_ptr) orelse return error.WeightNotMapped;
    const s = self.resolveWeight(scales_ptr) orelse return error.WeightNotMapped;
    const b = self.resolveWeight(biases_ptr) orelse return error.WeightNotMapped;

    try self.dev.matmulMlxQ4(
        out,
        w.buf, w.offset,
        s.buf, s.offset,
        b.buf, b.offset,
        acts, 0,
        m, k,
        group_size,
        scale_dtype == .bf16,
    );
}
```

- [ ] **Step 3: Run tests**

```sh
zig build test -Dmetal=true 2>&1 | tee /tmp/t.log | tail -20
grep -E "passed|FAIL" /tmp/t.log
```

Expected: pass.

- [ ] **Step 4: Commit**

```sh
git add src/runtime/metal_backend.zig
git commit -m "feat(metal): MetalBackend.matmulMlxQ4 host helper

Resolves weight/scales/biases pointers to (buf, offset) tuples via
resolveWeight, then dispatches the 4-buffer Metal kernel. Returns
WeightNotMapped if any pointer is outside known shards."
```

---

## Task 6: Route mlx_q4 through GPU in matmulRuntime

**Files:**
- Modify: `src/runtime/forward.zig`

The current dispatch (`forward.zig` lines 363–369) unconditionally calls the CPU MLX kernel. Add a GPU branch in front.

- [ ] **Step 1: Write a failing test in `forward.zig`**

Append:

```zig
test "matmulRuntime: GPU mlx_q4 matches CPU baseline within tolerance" {
    if (!@import("../metal/bridge.zig").metal_enabled) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // Same fixture as the metal_backend test, but compare to mlx.matmul on CPU.
    // ... build 2-shard backend, scratch acts buffer aliased into State ...
    // CPU expected:
    var cpu_out: [m]f32 = undefined;
    try @import("../kernels/mlx.zig").matmul(
        &cpu_out, weight_bytes, scales_bytes, biases_bytes, acts,
        m, k, 4, 64, .f16);

    // GPU via matmulRuntime:
    var gpu_out_slice = state.xb[0..m];
    try matmulRuntime(&pool, &be, gpu_out_slice, typed_tensor_mlxq4, state.xb2[0..k], m, k);

    for (cpu_out, gpu_out_slice) |c_v, g_v| {
        try std.testing.expectApproxEqAbs(c_v, g_v, 1e-3);
    }
}
```

- [ ] **Step 2: Add the GPU dispatch in `matmulRuntime`**

Replace the existing `if (q == .mlx_q4)` branch (forward.zig lines 363–370):

```zig
if (q == .mlx_q4) {
    const aux = w.mlx orelse return error.UnsupportedWeightType;

    // GPU path: 4-bit only, requires acts and out aliased into MetalBackend
    // scratch buffers (so they live in MTLBuffers we control).
    if (metal) |mb| if (aux.bits == 4) {
        if (mb.scratchForPtr(out.ptr)) |out_buf| {
            if (mb.scratchForPtr(acts.ptr)) |acts_buf| {
                mb.matmulMlxQ4(
                    out_buf,
                    w.bytes.ptr, aux.scales.ptr, aux.biases.ptr,
                    acts_buf,
                    m, k,
                    aux.bits, @intCast(aux.group_size),
                    aux.scale_dtype,
                ) catch {
                    // Fall through to CPU path below.
                    break :gpu;
                };
                return;
            }
        }
    };

    // CPU fallback: bits != 4, or weight pointers not in any wrapped shard,
    // or out/acts not in scratch.
    @import("../kernels/mlx.zig").matmul(
        out, w.bytes, aux.scales, aux.biases, acts,
        m, k, aux.bits, aux.group_size, aux.scale_dtype,
    ) catch return error.UnsupportedWeightType;
    return;
}
```

(Wrap the `if (metal) |mb| if (aux.bits == 4) { ... }` in a labeled block `gpu:` to allow `break :gpu;` for a clean fallback.)

- [ ] **Step 3: Run tests**

```sh
zig build test -Dmetal=true
```

Expected: pass — both CPU and GPU produce the same logits within 1e-3.

- [ ] **Step 4: Commit**

```sh
git add src/runtime/forward.zig
git commit -m "feat(forward): route mlx_q4 matmul to GPU when MetalBackend present

When acts and out alias MetalBackend scratch buffers and bits == 4,
dispatch via mb.matmulMlxQ4. Other widths and unmapped pointers fall
back to the CPU mlx kernel."
```

---

## Task 7: Wire MetalBackend through the HF/MLX CLI

**Files:**
- Modify: `src/main.zig`
- Modify: `src/runtime/server.zig`

- [ ] **Step 1: Read `runHfCli` (around lines 407–449) to identify the `State.init(gpa, &model, null)` call**

```sh
grep -n "State.init.*null" src/main.zig
```

- [ ] **Step 2: Construct MetalBackend.initFromHf alongside model**

Replace the relevant block in `runHfCli`:

```zig
var maybe_metal: ?vantablack.MetalBackend = blk: {
    if (!vantablack.metal.metal_enabled) break :blk null;
    break :blk vantablack.MetalBackend.initFromHf(gpa, &bundle, model.config) catch |err| {
        std.log.warn("MetalBackend.initFromHf failed: {s}; falling back to CPU", .{@errorName(err)});
        break :blk null;
    };
};
defer if (maybe_metal) |*mb| mb.deinit(gpa);

const metal_ptr: ?*vantablack.MetalBackend = if (maybe_metal) |*mb| mb else null;

var state = try vantablack.forward.State.init(gpa, &model, metal_ptr);
defer state.deinit(gpa);

// Pass metal_ptr into every step() call.
```

- [ ] **Step 3: Update `step()` invocations**

```sh
grep -n "vantablack\.forward\.step" src/main.zig
```

Each call site that previously passed `null` for the metal arg in the HF path now passes `metal_ptr`.

- [ ] **Step 4: Update `Server.initFromHf` (server.zig)**

```sh
grep -n "initFromHf" src/runtime/server.zig
```

Add an optional `metal: ?*MetalBackend` field to the server bundle, thread it through the per-request `step` calls. Pattern mirrors the existing GGUF `Server.init` which already accepts metal.

- [ ] **Step 5: Build + run a smoke generate**

```sh
zig build -Doptimize=ReleaseFast -Dmetal=true
./zig-out/bin/vantablack ~/models/tinyllama-mlx-4bit prompt 16 "Once upon a time"
```

Expected: coherent output. Watch `stderr` for the warn log — if it appears, MetalBackend.initFromHf is failing and the run is on CPU.

- [ ] **Step 6: Commit**

```sh
git add src/main.zig src/runtime/server.zig
git commit -m "feat(cli): construct MetalBackend.initFromHf for MLX models

runHfCli and Server.initFromHf now build a per-shard MetalBackend
when -Dmetal=true. Failure logs a warn and falls back to CPU."
```

---

## Task 8: Bit-equality smoke test against CPU baseline

**Files:**
- Create: `tests/mlx_gpu_smoke.zig` (or add to `src/runtime/forward.zig` test block)

End-to-end: load real MLX TinyLlama-1.1B-Chat-v1.0-4bit, run greedy decode for 32 tokens with metal off and metal on, assert token IDs are identical.

- [ ] **Step 1: Add the test (gated on env var)**

```zig
test "mlx-q4 GPU produces identical token ids to CPU on TinyLlama" {
    if (!@import("../metal/bridge.zig").metal_enabled) return error.SkipZigTest;
    const env_dir = std.posix.getenv("MLX_TEST_DIR") orelse return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var bundle = try vantablack.HfBundle.init(allocator, env_dir);
    defer bundle.deinit();

    var model = try vantablack.Model.initFromHf(allocator, &bundle);
    defer model.deinit(allocator);

    var pool = try vantablack.ThreadPool.init(allocator, 1);
    defer pool.deinit();

    const prompt_ids = [_]u32{ 1, 4874, 2501, 263, 931 }; // "Once upon a time"
    const N: usize = 32;

    // CPU run.
    var cpu_ids: [N]u32 = undefined;
    {
        var cache = try vantablack.KvCache.init(allocator, model.config);
        defer cache.deinit(allocator);
        var state = try vantablack.forward.State.init(allocator, &model, null);
        defer state.deinit(allocator);
        for (prompt_ids) |id| try vantablack.forward.step(&model, &state, &cache, &pool, null, id);
        var last: u32 = prompt_ids[prompt_ids.len - 1];
        for (0..N) |i| {
            try vantablack.forward.step(&model, &state, &cache, &pool, null, last);
            last = vantablack.argmax(state.logits);
            cpu_ids[i] = last;
        }
    }

    // GPU run.
    var gpu_ids: [N]u32 = undefined;
    {
        var be = try vantablack.MetalBackend.initFromHf(allocator, &bundle, model.config);
        defer be.deinit(allocator);
        var cache = try vantablack.KvCache.init(allocator, model.config);
        defer cache.deinit(allocator);
        var state = try vantablack.forward.State.init(allocator, &model, &be);
        defer state.deinit(allocator);
        for (prompt_ids) |id| try vantablack.forward.step(&model, &state, &cache, &pool, &be, id);
        var last: u32 = prompt_ids[prompt_ids.len - 1];
        for (0..N) |i| {
            try vantablack.forward.step(&model, &state, &cache, &pool, &be, last);
            last = vantablack.argmax(state.logits);
            gpu_ids[i] = last;
        }
    }

    for (cpu_ids, gpu_ids, 0..) |c, g, i| {
        std.testing.expectEqual(c, g) catch |e| {
            std.debug.print("mismatch at i={d}: cpu={d} gpu={d}\n", .{ i, c, g });
            return e;
        };
    }
}
```

- [ ] **Step 2: Run the test**

```sh
MLX_TEST_DIR=$HOME/models/tinyllama-mlx-4bit zig build test -Dmetal=true 2>&1 | tail -20
```

Expected: pass. If token IDs diverge, the CPU/GPU kernels disagree — likely a subtle dtype / bias-fold difference; bisect with smaller `N`.

- [ ] **Step 3: Commit**

```sh
git add src/runtime/forward.zig
git commit -m "test(mlx): GPU vs CPU bit-equality on TinyLlama 32-token decode

Gated on MLX_TEST_DIR env var. Asserts identical token IDs from
metal-on vs metal-off greedy decode."
```

---

## Task 9: Performance smoke + README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Measure tok/s with and without metal**

```sh
zig build -Doptimize=ReleaseFast -Dmetal=true
time ./zig-out/bin/vantablack ~/models/tinyllama-mlx-4bit prompt 128 "Once upon a time"
# Then build without metal:
zig build -Doptimize=ReleaseFast
time ./zig-out/bin/vantablack ~/models/tinyllama-mlx-4bit prompt 128 "Once upon a time"
```

Record: tok/s for each.

- [ ] **Step 2: Update README Status table + MLX section**

Change the line:

```
| MLX 4-bit MSL kernel                    | compiled + cached; runtime dispatch from forward.zig pending |
```

To:

```
| MLX 4-bit MSL kernel                    | shipped — runtime GPU dispatch wired; bit-equal to CPU on TinyLlama |
```

Update the "Caveats / not-yet" first bullet ("MLX 4-bit on GPU"): replace with a one-liner describing what was done (matmul on GPU, full-forward GPU path still Q8_0-GGUF only).

Update roadmap item 12 to drop the "runtime dispatch from forward.zig is the last bit needed".

- [ ] **Step 3: Commit**

```sh
git add README.md
git commit -m "docs: MLX-Q4 GPU matmul shipped"
```

---

## Self-Review Checklist

Run before declaring done:

- [ ] Every task ends with passing `zig build test -Dmetal=true`
- [ ] `zig build test` (no metal) still passes — CPU path untouched
- [ ] `zig build -Doptimize=ReleaseSmall` still under 5 MB stripped
- [ ] GGUF Q8_0 path still hits full-GPU forward (untouched modelGpuEligible predicate)
- [ ] Token IDs from `prompt 32 "Once upon a time"` on TinyLlama-MLX-4bit are identical to a known-good capture (record before Task 1, replay after Task 8)
- [ ] No new `extern "C"` outside `src/metal/`

## Out of Scope (call out in PR description)

- Full GPU forward for MLX models (rmsnorm/RoPE/attn on GPU). Requires segment-API extensions for MLX scales/biases — separate plan.
- 2/3/5/6/8-bit MLX widths on GPU. CPU path remains; trivial extension once 4-bit is proven.
- Async pipelining across layers (Plan A).
- MLX-Q4 LM head on GPU. Logits matmul currently routed through the same matmulRuntime; should "just work" via the new branch but verify in Task 8 token-equality test.

---

# Follow-on Plans (summaries — to be expanded)

These are the remaining items from the README's limitations + roadmap. Each gets its own dated plan file when picked up. Listed in recommended execution order.

## Plan B: K-quant MSL kernels (Q4_K / Q5_K / Q6_K on GPU)

**Why:** Mixed-quant GGUF models (Q4_K_M, Q4_K_S) currently fall back to per-op CPU+GPU. Adds GPU coverage for the most-shipped quant family.

**Scope:**
- New MSL kernels: `matmul_q4_k`, `matmul_q5_k`, `matmul_q6_k` in `src/metal/bridge.m`. Reference: `src/kernels/simd.zig` block layouts.
- Segment API additions: `vtb_metal_segment_matmul_q4_k`, etc.
- Zig wrappers in `bridge.zig`.
- Extend `modelGpuEligible(m)` in `forward.zig` to permit K-quant projections (likely split into per-quant eligibility tables).
- Per-quant unit tests against CPU `simd.matmulQ4_K` etc.
- Integration test: TinyLlama Q4_K_M end-to-end, full-forward on GPU, identical to CPU.

**Risk:** K-quant block layout (super-blocks of 256 elements with sub-block scales) is more intricate than Q8_0 — kernel correctness needs careful unit testing first.

## Plan A: GPU async pipelining + weight fusion

**Why:** Closes the gap to llama.cpp (~65 → ~90 tok/s expected). Roadmap items 8 + 9.

**Scope:**
- Drop `waitUntilCompleted` per layer; chain command buffers with `addCompletedHandler` or rely on Metal's implicit serial queue.
- Pre-concatenate `attn_q/k/v` into a single `attn_qkv` weight tensor at `Model.init` (and likewise `ffn_gate` + `ffn_up`). One fused matmul per group.
- Update `Model.init` + `gpuLayerStep` to emit fused matmuls.
- Memory cost: ~660 MB extra for TinyLlama Q8_0 — flag as a build option `-Dweight_fusion=true` initially.

**Risk:** Async pipelining can mask correctness bugs (race on KV cache write vs. next-layer read). Need a careful synchronization model — likely one segment per layer plus an explicit fence on the KV-cache buffer between layers.

## Plan D: Prompt prefill batching

**Why:** Long prompts currently cost N forward passes. Batching N tokens in one forward gives ~N× speedup on prefill.

**Scope:**
- Generalize `forward.step` into `forward.steps(token_ids: []const u32)`.
- Add a batch dim to QKV / FFN matmuls: matmul becomes (M, K) × (K, B) instead of (M, K) × (K, 1). New MSL kernels `matmul_q8_0_batched`, etc.
- KV cache writes become slab-wide instead of single-slot.
- Attention scores grow to `n_heads × seq × B`.
- Server: detect prefill chunk on first request, dispatch in one batched forward.
- CLI: same change in `prompt`/`chat`.

**Risk:** Memory growth for `attn_scores` buffer scales with B. Need to size shared MTLBuffer accordingly. CPU path generalization is straightforward but extensive.

## Plan E: Tiktoken-style byte-level BPE

**Why:** Llama 3+, GPT-2, GPT-J, most modern OSS — all use byte-level BPE with a Unicode-class regex pre-tokenizer. Currently unsupported.

**Scope:**
- New module `src/runtime/tokenizer_bpe.zig` (or extend existing `tokenizer.zig` with a polymorphic `Encoder` interface).
- Implement GPT-2 byte→Unicode-string mapping (the 256-char alphabet).
- Implement Unicode property regex pre-tokenizer split: `\p{L}+ | \p{N}+ | [^\s\p{L}\p{N}]+ | \s+`. Pure-Zig — port a minimal Unicode general-category table.
- Detect tokenizer flavor from `tokenizer.json::pre_tokenizer` field; dispatch encoder accordingly.
- Tests: round-trip on Llama-3 vocab + GPT-2 vocab against published reference token sequences.

**Risk:** Pure-Zig Unicode regex is the load-bearing piece. Could vendor a small `unicode-regex` table or hand-roll. Avoid pulling in PCRE.

## Plan F: Multi-architecture support (Mistral / Phi / Gemma)

**Why:** Currently Llama-only. README limitation #3.

**Scope:**
- `LlamaConfig` becomes `ArchConfig` (tagged union over Llama / Mistral / Phi / Gemma).
- Per-arch metadata key tables (e.g. Mistral has `sliding_window`, Gemma has `hidden_activation`).
- Tensor-name remap in `Model.init` for each arch family.
- Per-arch forward-pass tweaks: Mistral sliding-window attention, Gemma `gelu_approx`, Phi parallel-block layout.
- Smoke tests: one model per arch family.

**Risk:** Phi-style parallel residual blocks change the forward graph shape — touches `gpuLayerStep` segment encoding too.

## Plan G: CPU SIMD wins (NEON sdot + Apple AMX)

**Why:** Roadmap items 12 + 13. Improves the no-Metal path (Linux arm64, Windows, x86) and gives a low-power CPU fallback on Apple.

**Scope:**
- NEON `sdot` int8 dot via Zig inline-asm or `@Vector` intrinsics in `src/kernels/simd.zig`. Q8_0 inner loop becomes 4× wider.
- Apple AMX matmul via inline-asm against the documented (private) AMX instruction set; gated on `builtin.cpu.arch == .aarch64 and is_apple`. Use only for the largest matmul (`output.weight`) where setup cost amortizes.
- Benchmark harness in `bench/` directory.

**Risk:** AMX is an undocumented coprocessor — instruction set has been reverse-engineered but is not stable across `M1/M2/M3/M4`. Treat as best-effort and feature-flag.

## Plan H: Server polish

**Why:** Limitations: serial requests, placeholder timestamps, missing `eval_count`/`total_duration`, yielding-not-parking workers.

**Scope:**
- Real timestamps via `std.time.timestamp` rendered as ISO-8601.
- SHA-256 of model file (one-shot at server startup, cached) for `digest`.
- Per-request token counters threaded through `forward.step` return value.
- Optional concurrency: per-request `KvCache` + scratch state pool (N concurrent requests, one MetalBackend shared). Big change; could be a separate plan.
- Worker parking: replace `spinLoopHint`+`yield` with `std.Thread.Futex` (Zig 0.16 std exposes it through `std.atomic`). Event-driven wake on dispatch.

**Risk:** Concurrent requests sharing one MetalBackend means serializing GPU dispatches anyway — may not be worth it for single-GPU, single-user. Instead document multi-process pattern.

## Plan I: Verification backlog

**Why:** TQ2_0 untested on real weights, multi-shard MLX never inference-smoked, 7B+ Llama models never smoked.

**Scope:**
- Find a public TQ2_0 GGUF (or generate one from a 1.58-bit BitNet checkpoint via a small Python tool); add `tests/tq2_0_smoke.zig`.
- Download a sharded MLX model (e.g. `mlx-community/Mistral-7B-Instruct-v0.2-4bit`); add `tests/mlx_multi_shard_smoke.zig`.
- 7B Llama Q4_K_M smoke; 13B if disk permits.
- Each gated on env var so CI stays fast.

**Risk:** Disk + bandwidth. Skip or move to a `nightly` test target.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-03-mlx-q4-gpu-dispatch.md`. Two execution options:

1. **Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
