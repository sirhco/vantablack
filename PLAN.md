# Vantablack — Task 1: The Foundation

## Context

`vantablack` is a zero-footprint, pure-Zig (0.16+) local LLM inference engine. Goal: lowest possible memory overhead so 70B models — including 1.58-bit ternary nets — run on modest hardware without `llama.cpp` or any C++ runtime. The repo at `/Users/chrisolson/development/github/vantablack` is currently a fresh `zig init` scaffold (`build.zig`, `build.zig.zon`, `src/main.zig`, `src/root.zig`). Zig 0.16.0 is installed at `/opt/homebrew/bin/zig`.

Task 1 ships the spine: zero-copy `mmap`-backed weight loading, a GGUF v3 parser exposing a typed tensor catalog, and a comptime kernel dispatcher with three registered quant paths. Inference loop comes later. Acceptance: a CLI that prints the tensor catalog of any GGUF file, and a stripped ReleaseSmall binary under 5 MB.

User-confirmed choices (asked & answered): literal `std.posix.mmap` (not `std.Io.File.MemoryMap`); padded-column CLI output; default `zig build` stays Debug; macOS arm64 only for Task 1.

## Files — keep, modify, delete, create

| Path                                | Action  | Notes |
|-------------------------------------|---------|-------|
| `build.zig`                         | modify  | Path swap to `vantablack.zig`; add strip + single_threaded + omit_frame_pointer + error_tracing tuning per optimize level |
| `build.zig.zon`                     | keep    | No changes |
| `src/main.zig`                      | rewrite | Open argv[1], `ModelMapper.init`, iterate catalog, print padded columns. Keep existing `simple test` and `fuzz example` blocks. |
| `src/root.zig`                      | DELETE  | Replaced by `src/vantablack.zig` (matches package name; explicit module identity) |
| `src/vantablack.zig`                | CREATE  | Public module index — re-export `core/`, `kernels/`, top-level types |
| `src/core/mapper.zig`               | CREATE  | `ModelMapper` (init / tensorSlice / deinit), owns mmap + arena |
| `src/core/parser.zig`               | CREATE  | `GGUFParser`, `GgmlType`, `MetaValue`, `MetaKv`, `TensorDesc`, `Catalog`, block-info table |
| `src/kernels/comptime_gen.zig`      | CREATE  | `QuantType`, `Kernel` typedef, `dispatch(comptime q)` |
| `src/kernels/simd.zig`              | CREATE  | Five stub kernels (`@panic("todo")`) + real `dot_f32` using `@Vector(8, f32)` |

`mkdir src/core src/kernels` is required before file creation.

## build.zig edits

1. Swap `b.path("src/root.zig")` → `b.path("src/vantablack.zig")` on line 38.
2. Compute per-mode flags from `optimize`:
   - `is_release = optimize != .Debug`
   - `strip = is_release`
   - `single_threaded = is_release` (no threads in Task 1; drops TLS scaffolding from libstd)
   - `omit_frame_pointer = is_release`
   - `error_tracing: ?bool = if (optimize == .ReleaseSmall) false else null`
3. Pass these to BOTH `b.addModule("vantablack", ...)` and the exe's `b.createModule(...)`.
4. Leave `link_libc` null (mmap works through `std.posix` syscalls without libc on macOS).
5. Default optimize stays Debug (per user choice). Verification step explicitly passes `-Doptimize=ReleaseSmall`.
6. `zig build test` keeps working — `mod_tests` now pulls tests transitively from `vantablack.zig` → `core/*` + `kernels/*`. `exe_tests` still tests `main.zig`.

## Component details

### `src/core/mapper.zig` — ModelMapper

```
pub const ModelMapper = struct {
    arena: std.heap.ArenaAllocator,                     // owns catalog allocations
    fd: std.posix.fd_t,                                 // owned; closed in deinit
    mapped: []align(std.heap.page_size_min) const u8,   // owned; munmap'd in deinit
    catalog: parser.Catalog,
    data_segment_start: u64,

    pub fn init(allocator: Allocator, abs_path: []const u8) !ModelMapper;
    pub fn deinit(self: *ModelMapper) void;
    pub fn tensorSlice(self: *const ModelMapper, name: []const u8) ![]const u8;
};
```

Init order (each step has matching `errdefer`):
1. `arena = ArenaAllocator.init(allocator)`
2. `fd = try std.posix.openat(std.posix.AT.FDCWD, path_z, .{ .ACCMODE = .RDONLY }, 0)`. Path must be NUL-terminated (allocator-backed sentinel slice).
3. `stat = try std.posix.fstat(fd)`; reject `stat.size < 24` with `error.FileTooSmall`; reject `stat.size == 0` with `error.FileEmpty`.
4. `mapped = try std.posix.mmap(null, @intCast(stat.size), std.posix.PROT.READ, .{ .TYPE = .PRIVATE }, fd, 0)`
5. Validate magic (`mapped[0..4] == "GGUF"`) and version (2 or 3).
6. `catalog = try parser.parseHeader(arena.allocator(), mapped)` — fills descs + metadata + data_segment_start + alignment.

Deinit (reverse): `std.posix.munmap(mapped); std.posix.close(fd); arena.deinit();`

`tensorSlice(name)`: linear scan `catalog.descs` for `name` (Task 1; Task 2+ may add a hashmap). Compute `abs_offset = data_segment_start + desc.rel_offset`. Bounds-check `abs_offset + desc.size_bytes <= mapped.len`. Return `mapped[abs_offset..][0..desc.size_bytes]`.

### `src/core/parser.zig` — GGUFParser

GGUF v3 file layout (little-endian):
```
magic u32 = 'GGUF'  | version u32  | tensor_count u64  | metadata_kv_count u64
metadata_kv[]: { key:gguf_string, value_type:u32, value:gguf_value }
tensor_info[]: { name:gguf_string, n_dims:u32, dims:[n_dims]u64, ggml_type:u32, rel_offset:u64 }
[ pad to general.alignment (default 32; override via metadata key "general.alignment":u32) ]
[ tensor data ]
```

Versions accepted: 2 and 3 (byte-compatible for what we read). Reject everything else.

`gguf_string = { length:u64, bytes:[length]u8 }`. No NUL. UTF-8 expected; do not validate strictly.

Twelve metadata value types must be handled (u8, i8, u16, i16, u32, i32, f32, bool, string, array, u64, i64, f64). Real-world GGUF uses all of them. ARRAY recurses; depth ≥ 4 acceptable. ARRAY-of-STRING is the painful path — store array payloads as raw `[]const u8` slice into the mmap plus an `elem_type` cursor; only iterate when consumer asks.

`GgmlType` is `enum(u32) { f32=0, f16=1, q4_0=2, q4_1=3, q5_0=6, q5_1=7, q8_0=8, q8_1=9, q2_k=10, q3_k=11, q4_k=12, q5_k=13, q6_k=14, q8_k=15, iq2_xxs=16, iq2_xs=17, iq3_xxs=18, iq1_s=19, iq4_nl=20, iq3_s=21, iq2_s=22, iq4_xs=23, i8=24, i16=25, i32=26, i64=27, f64=28, iq1_m=29, bf16=30, tq1_0=31, tq2_0=32, _ }`. Open enum (`_`) for forward-compat; unknown future types do not hard-fail catalog enumeration.

Per-element / per-block size table (source: `ggml-quants.h` in ggml repo):

| ggml_type | block elements | bytes per block |
|-----------|----------------|-----------------|
| F32       | 1              | 4               |
| F16       | 1              | 2               |
| BF16      | 1              | 2               |
| Q8_0      | 32             | 34              |
| Q4_K      | 256            | 144             |
| TQ1_0     | 256            | 54              |
| TQ2_0     | 256            | 66              |

Encode as `pub const block_info: []const BlockInfo` indexed by `GgmlType`. Compute `size_bytes = (numel / block_elems) * bytes_per_block`. Assert `numel % block_elems == 0` (GGUF guarantees this for tensors using these quants).

Returned types:
```
pub const TensorDesc = struct {
    name: []const u8,                  // borrows from mmap
    dims: []const u64,                 // arena-copied (alignment-safe via std.mem.readInt)
    ggml_type: GgmlType,
    rel_offset: u64,                   // relative to data_segment_start
    size_bytes: u64,
};
pub const Catalog = struct {
    descs: []TensorDesc,               // arena-owned
    metadata: []const MetaKv,          // arena-owned (slice container)
    data_segment_start: u64,
    alignment: u32,
    pub fn find(self: Catalog, name: []const u8) ?*const TensorDesc;
};
```

Zero-copy: `name`, `MetaKv.key`, string MetaValues, raw-array MetaValue payloads slice into mmap. `dims[]` is arena-copied (GGUF doesn't guarantee u64 alignment of tensor_info records — use `std.mem.readInt(u64, bytes[off..][0..8], .little)`, not `@ptrCast`). Numeric metadata scalars held by-value in the union — no allocation.

After parsing all `tensor_info` records, round the byte cursor up to `alignment` (default 32; override from `general.alignment` if present). Store as `data_segment_start`.

### `src/kernels/comptime_gen.zig` — KernelDispatcher

```zig
pub const QuantType = enum { f32, f16, q8_0, q4_k, ternary158 };

pub const Kernel = *const fn (
    out: []f32,
    weights: []const u8,
    acts: []const f32,
    m: usize,
    k: usize,
) void;

pub fn dispatch(comptime q: QuantType) Kernel {
    return switch (q) {
        .f32        => &@import("simd.zig").matmul_f32,
        .f16        => &@import("simd.zig").matmul_f16,
        .q8_0       => &@import("simd.zig").matmul_q8_0,
        .q4_k       => &@import("simd.zig").matmul_q4_k,
        .ternary158 => &@import("simd.zig").matmul_ternary158,
    };
}
```

For Task 1, every `matmul_*` body is `@panic("kernel not yet implemented: " ++ @tagName(q))`. Future inner loop pattern:
```
switch (layer.quant) {
    inline .q8_0, .q4_k, .ternary158 => |q| {
        const kernel = comptime dispatch(q);
        kernel(scratch, layer.weights, acts, layer.m, layer.k);
    },
    else => @panic("unreachable"),
}
```
The inline-prong + `comptime dispatch(q)` emits one specialized direct call per quant variant. Per-element inner loop sees zero branching. (One per-layer tag check remains — that is the correct interpretation of "comptime dispatch" when the layer schedule is runtime-built.)

### `src/kernels/simd.zig`

Five stub kernels matching the `Kernel` signature plus one real function as `@Vector` smoke:
```zig
pub fn dot_f32(a: []const f32, b: []const f32) f32 { /* @Vector(8, f32) reduce */ }
```
Used in unit tests to prove SIMD codegen wired correctly.

### `src/vantablack.zig` — public module index

Re-exports for downstream consumers:
```zig
pub const mapper = @import("core/mapper.zig");
pub const parser = @import("core/parser.zig");
pub const kernels = @import("kernels/comptime_gen.zig");
pub const simd = @import("kernels/simd.zig");

pub const ModelMapper = mapper.ModelMapper;
pub const Catalog = parser.Catalog;
pub const TensorDesc = parser.TensorDesc;
pub const GgmlType = parser.GgmlType;
pub const QuantType = kernels.QuantType;
pub const Kernel = kernels.Kernel;
pub const dispatch = kernels.dispatch;
```

### `src/main.zig` — CLI

Keep the Zig 0.16 entry point shape (`pub fn main(init: std.process.Init) !void`), `init.arena.allocator()`, `init.minimal.args.toSlice`, `init.io`, and the existing `std.Io.File.Writer` pattern. Keep the `simple test` and `fuzz example` blocks at the bottom (free coverage). Body:

1. Parse args. If `args.len < 2`: print usage to stderr, return `error.MissingPath`.
2. Resolve `args[1]` to absolute path via `std.fs.path.resolve(arena, &.{args[1]})` if not already absolute.
3. `var mapper = try vantablack.ModelMapper.init(arena, abs_path); defer mapper.deinit();`
4. Compute column widths from `mapper.catalog.descs` (max name length, max dims-rendered length).
5. For each desc: print padded line `name  [d0, d1, ...]  GGML_TYPE  offset=0x...  size=N` to stdout.
6. `try stdout_writer.flush();`

Example output:
```
token_embd.weight        [32000, 2048]      Q4_K        offset=0x000020   size=18874368
output_norm.weight       [2048]             F32         offset=0x1200020  size=8192
output.weight            [32000, 2048]      Q6_K        offset=0x1202020  size=53739520
```

## Lifetime / allocator strategy

One `ArenaAllocator` per `ModelMapper`, backed by the parent allocator passed into `init`. Catalog lifetimes nest perfectly inside `ModelMapper` lifetime — textbook arena fit. No per-tensor frees, no double-free risk, one bulk free at `deinit`. Future per-forward-pass scratch buffers (Task 2+) will use the parent allocator directly so inference scratch doesn't tie to model lifetime. **No global state** — allocator, fd, and mmap are all on the struct.

## Verification

1. **Build all three modes:**
   ```
   /opt/homebrew/bin/zig build
   /opt/homebrew/bin/zig build -Doptimize=ReleaseFast
   /opt/homebrew/bin/zig build -Doptimize=ReleaseSmall
   ```

2. **Tests pass:** `/opt/homebrew/bin/zig build test`. Suite must cover:
   - `parser.zig`: hand-build a minimal valid GGUF v3 byte buffer in a `test` body (magic + version=3 + tensor_count=1 + kv_count=2 + STRING `general.architecture="test"` + U32 `general.alignment=32` + tensor `"w"` n_dims=2 dims=[4,4] ggml_type=F32 rel_offset=0 + 32-byte align padding + 64 bytes of f32 zeros). Parse it; assert `descs.len == 1`, `descs[0].name == "w"`, `descs[0].dims == [_]u64{4,4}`, `descs[0].size_bytes == 64`, `metadata.len == 2`, `alignment == 32`.
   - `mapper.zig`: write that same byte array to a tmpfile, `ModelMapper.init`, `tensorSlice("w")`, assert length 64 and all-zero, deinit, delete tmpfile.
   - `comptime_gen.zig`: `try std.testing.expect(@TypeOf(dispatch(.q8_0)) == Kernel);` (compile-time signature lock).
   - `simd.zig`: `dot_f32` against hand-computed value, proves `@Vector(8, f32)` codegens.

3. **Binary size under 5 MB:**
   ```
   /opt/homebrew/bin/zig build -Doptimize=ReleaseSmall
   ls -lh zig-out/bin/vantablack
   ```
   If over 5 MB, escalate in order: confirm `single_threaded` applied; confirm `error_tracing=false` applied; try `-Doptimize=ReleaseFast` (sometimes smaller on tiny binaries); add `omit_frame_pointer=true` everywhere; `strip -x zig-out/bin/vantablack` (removes local symbols); last resort `use_llvm=true`.

4. **CLI smoke (manual, requires a `.gguf`):**
   ```
   /opt/homebrew/bin/zig build -Doptimize=ReleaseFast
   ./zig-out/bin/vantablack ~/path/to/model.gguf
   ```
   Suggested test files (NOT committed): `karpathy/tinyllamas` 15M (~25 MB) for fast iteration; TinyLlama-1.1B-Chat Q4_K_M (~670 MB) for Q4_K + Q6_K + F32 coverage.

5. **Git hygiene:** `git status` should show `build.zig` modified, `src/main.zig` modified, `src/root.zig` deleted, plus new files `src/vantablack.zig`, `src/core/{mapper,parser}.zig`, `src/kernels/{simd,comptime_gen}.zig`.

## Reference docs

- GGUF v3 spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- GGML quant block sizes (source of truth for `block_info` table): `ggml-quants.h` in the ggml repo.
- Zig 0.16 stdlib: `std.posix.{openat,fstat,mmap,munmap,close}`, `std.heap.ArenaAllocator`, `std.heap.page_size_min`, `std.mem.readInt`, `std.process.Init`, `std.Io.File.Writer`, `@Vector`.
