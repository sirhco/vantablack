//! C ABI shim for vantablack.
//!
//! Exposes the minimum surface a Swift / Kotlin / generic C host needs to
//! run on-device inference: open a model, init a per-session state, stream
//! tokens, forward memory/thermal signals. Everything is opaque pointers
//! so future field changes don't break the ABI.
//!
//! Header is hand-maintained in `include/vantablack.h` — keep that file
//! in lockstep with the symbols below.
//!
//! Scope (v1): GGUF mmap path only. HuggingFace / MLX directory bundles
//! are a follow-up — they need an `HfBundle` lifetime + different
//! tokenizer init. The current header reflects the GGUF-only contract.

const std = @import("std");
const builtin = @import("builtin");

const vantablack = @import("vantablack.zig");

const Allocator = std.mem.Allocator;
const Io = std.Io;

// Single-threaded Io implementation — sufficient for synchronous model
// open + per-token generation. Avoids dragging Threaded's worker pool
// into the static library.
fn defaultIo() Io {
    return std.Io.Threaded.global_single_threaded.io();
}

// ============================================================================
// Opaque types
// ============================================================================

pub const VtbModel = extern struct {
    _opaque: [0]u8 = .{},
};

pub const VtbState = extern struct {
    _opaque: [0]u8 = .{},
};

// Internal Zig-side bookkeeping behind each opaque pointer. The backing
// `union(enum)` lets the same handle host either a GGUF mmap (single
// file) or an HF/MLX bundle (directory with shards + tokenizer.json).
// Downstream `vtb_state_init` / `vtb_generate_stream` are uniform over
// both — the Backend vtable already hides the difference.
const Backing = union(enum) {
    gguf: vantablack.ModelMapper,
    hf: vantablack.hf_loader.HfBundle,
};

const ModelHandle = struct {
    gpa: Allocator,
    arena_state: std.heap.ArenaAllocator,
    backing: Backing,
    model: vantablack.model.Model,
    metal: ?vantablack.MetalBackend,
    tokenizer: vantablack.Tokenizer,
    pool: *vantablack.ThreadPool,
};

const StateHandle = struct {
    parent: *ModelHandle,
    state: vantablack.forward.State,
    cache: vantablack.KvCache,
    sampler: vantablack.Sampler,
    backend_cpu: vantablack.CpuBackend,
    backend_metal: ?vantablack.MetalDispatch,
};

// Process-wide pressure hub. Shared by every active session — matches
// the OS notification model (one source, fan-out).
var g_hub: vantablack.PressureHub = .{};

// Global allocator for the C-API path. Apps that need a custom allocator
// should call the Zig API directly; the C ABI keeps the surface small.
// `SmpAllocator` is the Zig-0.16 thread-safe general-purpose allocator;
// safe to call from any OS thread including the iOS notification queue.
fn gpaAllocator() Allocator {
    return std.heap.smp_allocator;
}

// ============================================================================
// Status / error codes
// ============================================================================

pub const VTB_OK: c_int = 0;
pub const VTB_ERR_INVALID_ARG: c_int = -1;
pub const VTB_ERR_FILE_OPEN: c_int = -2;
pub const VTB_ERR_PARSE: c_int = -3;
pub const VTB_ERR_OOM: c_int = -4;
pub const VTB_ERR_METAL: c_int = -5;
pub const VTB_ERR_DISPATCH: c_int = -6;
pub const VTB_ERR_STREAM_ABORTED: c_int = -7;
pub const VTB_ERR_UNSUPPORTED: c_int = -8;

// ============================================================================
// Model lifecycle
// ============================================================================

/// Open a GGUF file at `path` (null-terminated). On success returns a
/// non-null `VtbModel *`. On failure returns null; check stderr for the
/// underlying error. `metal_enabled` selects the GPU backend when the
/// library was built with `-Dmetal=true`; otherwise it's ignored.
export fn vtb_model_open(path: [*:0]const u8, metal_enabled: u8) callconv(.c) ?*VtbModel {
    return vtbModelOpenImpl(std.mem.span(path), metal_enabled != 0) catch null;
}

fn vtbModelOpenImpl(path: []const u8, want_metal: bool) !*VtbModel {
    const gpa = gpaAllocator();
    var handle = try gpa.create(ModelHandle);
    errdefer gpa.destroy(handle);

    handle.gpa = gpa;
    handle.arena_state = std.heap.ArenaAllocator.init(gpa);
    errdefer handle.arena_state.deinit();

    // C ABI requires absolute paths — callers know their app's bundle
    // root (NSBundle.bundleURL on iOS, ApplicationDir on Android) and
    // resolution is simpler in the host language than here.
    if (!std.fs.path.isAbsolute(path)) return error.PathNotAbsolute;

    var mapper = try vantablack.ModelMapper.init(gpa, defaultIo(), path);
    errdefer mapper.deinit();
    handle.backing = .{ .gguf = mapper };

    handle.model = try vantablack.model.Model.init(gpa, &handle.backing.gguf);
    errdefer handle.model.deinit(gpa);

    handle.metal = blk: {
        if (!want_metal) break :blk null;
        if (!vantablack.metal.metal_enabled) break :blk null;
        break :blk vantablack.MetalBackend.init(gpa, &handle.backing.gguf, handle.model.config) catch null;
    };
    errdefer if (handle.metal) |*mb| mb.deinit(gpa);
    if (handle.metal) |*mb| mb.attachFusedWeights(gpa, &handle.model) catch {};

    handle.tokenizer = try vantablack.Tokenizer.init(gpa, handle.backing.gguf.catalog);
    errdefer handle.tokenizer.deinit(gpa);

    // Metal does the heavy lifting; one worker is plenty in that mode.
    const threads: usize = if (handle.metal != null) 1 else 0;
    handle.pool = try vantablack.ThreadPool.init(gpa, threads);

    return @ptrCast(handle);
}

/// Open an HF / MLX directory bundle (config.json + safetensors shards +
/// tokenizer.json). Same surface as `vtb_model_open` but for the multi-
/// file directory format `mlx-community/*` repos ship. `dir_path` must
/// be an absolute path.
export fn vtb_model_open_dir(dir_path: [*:0]const u8, metal_enabled: u8) callconv(.c) ?*VtbModel {
    return vtbModelOpenDirImpl(std.mem.span(dir_path), metal_enabled != 0) catch null;
}

fn vtbModelOpenDirImpl(dir_path: []const u8, want_metal: bool) !*VtbModel {
    const gpa = gpaAllocator();
    var handle = try gpa.create(ModelHandle);
    errdefer gpa.destroy(handle);

    handle.gpa = gpa;
    handle.arena_state = std.heap.ArenaAllocator.init(gpa);
    errdefer handle.arena_state.deinit();

    if (!std.fs.path.isAbsolute(dir_path)) return error.PathNotAbsolute;

    var bundle = try vantablack.hf_loader.HfBundle.init(gpa, defaultIo(), dir_path);
    errdefer bundle.deinit();
    handle.backing = .{ .hf = bundle };

    // Parse config.json. HfConfig holds slices into the supplied
    // allocator's arena — use the handle's arena so the cfg lifetime
    // matches the handle. (Model.initFromHf only reads scalar fields,
    // but model_type / quantization slices remain alive.)
    const cfg = try vantablack.hf_config.parse(handle.arena_state.allocator(), handle.backing.hf.config_json);

    handle.model = try vantablack.model.Model.initFromHf(gpa, &handle.backing.hf, cfg);
    errdefer handle.model.deinit(gpa);

    handle.metal = blk: {
        if (!want_metal) break :blk null;
        if (!vantablack.metal.metal_enabled) break :blk null;
        break :blk vantablack.MetalBackend.initFromHf(gpa, &handle.backing.hf, handle.model.config) catch null;
    };
    errdefer if (handle.metal) |*mb| mb.deinit(gpa);
    if (handle.metal) |*mb| mb.attachFusedWeights(gpa, &handle.model) catch {};

    const tok_bytes = handle.backing.hf.tokenizer_json orelse return error.NoTokenizerJson;
    handle.tokenizer = try vantablack.Tokenizer.initFromHfJson(gpa, tok_bytes);
    errdefer handle.tokenizer.deinit(gpa);

    const threads: usize = if (handle.metal != null) 1 else 0;
    handle.pool = try vantablack.ThreadPool.init(gpa, threads);

    return @ptrCast(handle);
}

export fn vtb_model_close(m: ?*VtbModel) callconv(.c) void {
    const handle = castModel(m) orelse return;
    const gpa = handle.gpa;
    handle.pool.deinit(gpa);
    handle.tokenizer.deinit(gpa);
    if (handle.metal) |*mb| mb.deinit(gpa);
    handle.model.deinit(gpa);
    switch (handle.backing) {
        .gguf => |*mapper| mapper.deinit(),
        .hf => |*bundle| bundle.deinit(),
    }
    handle.arena_state.deinit();
    gpa.destroy(handle);
}

// ============================================================================
// Sampler config (mirrored to C)
// ============================================================================

pub const VtbSamplerConfig = extern struct {
    temperature: f32 = 0.0,
    top_k: usize = 0,
    top_p: f32 = 0.0,
    seed: u64 = 0,
};

// ============================================================================
// State lifecycle
// ============================================================================

export fn vtb_state_init(
    m: ?*VtbModel,
    sampler_cfg: *const VtbSamplerConfig,
) callconv(.c) ?*VtbState {
    const parent = castModel(m) orelse return null;
    return vtbStateInitImpl(parent, sampler_cfg.*) catch null;
}

fn vtbStateInitImpl(parent: *ModelHandle, cfg: VtbSamplerConfig) !*VtbState {
    const gpa = parent.gpa;
    var h = try gpa.create(StateHandle);
    errdefer gpa.destroy(h);
    h.parent = parent;
    h.backend_metal = null;

    const metal_ptr: ?*vantablack.MetalBackend = if (parent.metal) |*mb| mb else null;

    h.state = try vantablack.forward.State.init(gpa, &parent.model, metal_ptr);
    errdefer h.state.deinit(gpa);

    h.cache = try vantablack.KvCache.init(
        gpa,
        parent.model.config.n_layers,
        parent.model.config.n_kv_heads,
        parent.model.config.head_dim,
        parent.model.config.max_seq,
        metal_ptr,
    );
    errdefer h.cache.deinit(gpa);

    h.sampler = try vantablack.Sampler.init(gpa, .{
        .temperature = cfg.temperature,
        .top_k = cfg.top_k,
        .top_p = cfg.top_p,
        .seed = cfg.seed,
    }, parent.model.config.vocab_size);

    h.backend_cpu = .{ .pool = parent.pool, .metal_fallback = metal_ptr };
    if (metal_ptr) |mb| h.backend_metal = vantablack.MetalDispatch.init(mb);

    return @ptrCast(h);
}

export fn vtb_state_deinit(s: ?*VtbState) callconv(.c) void {
    const h = castState(s) orelse return;
    const gpa = h.parent.gpa;
    h.sampler.deinit(gpa);
    h.cache.deinit(gpa);
    h.state.deinit(gpa);
    gpa.destroy(h);
}

// ============================================================================
// Streaming generation
// ============================================================================

pub const VtbTokenCb = *const fn (
    ctx: *anyopaque,
    token_id: u32,
    piece_bytes: [*]const u8,
    piece_len: usize,
    is_final: u8,
) callconv(.c) u8;

/// Generate up to `max_tokens` tokens from `prompt` (UTF-8 bytes, no null
/// terminator required — pass explicit length). Calls `cb` once per
/// generated token. Returns 0 on success, negative on error.
export fn vtb_generate_stream(
    s: ?*VtbState,
    prompt_bytes: [*]const u8,
    prompt_len: usize,
    max_tokens: usize,
    cb: VtbTokenCb,
    cb_ctx: ?*anyopaque,
) callconv(.c) c_int {
    const h = castState(s) orelse return VTB_ERR_INVALID_ARG;
    const ctx: *anyopaque = cb_ctx orelse @as(*anyopaque, @ptrFromInt(@intFromPtr(&dummy_ctx)));

    const backend: vantablack.Backend = if (h.backend_metal) |*mb_disp|
        mb_disp.backend()
    else
        h.backend_cpu.backend();

    const prompt = prompt_bytes[0..prompt_len];

    vantablack.generateStream(
        h.parent.gpa,
        &h.parent.model,
        &h.parent.tokenizer,
        &h.sampler,
        backend,
        &h.state,
        &h.cache,
        prompt,
        max_tokens,
        cb,
        ctx,
    ) catch |err| return mapStreamErr(err);

    return VTB_OK;
}

var dummy_ctx: u8 = 0;

fn mapStreamErr(err: anyerror) c_int {
    return switch (err) {
        error.OutOfMemory => VTB_ERR_OOM,
        error.TokenOutOfRange => VTB_ERR_INVALID_ARG,
        error.KvCacheFull => VTB_ERR_STREAM_ABORTED,
        error.UnsupportedWeightType => VTB_ERR_UNSUPPORTED,
        else => VTB_ERR_DISPATCH,
    };
}

// ============================================================================
// Pressure / thermal hooks
// ============================================================================

/// `level`: 0 = normal, 1 = warning, 2 = critical. Out-of-range clamps
/// to critical. Safe to call from any thread.
export fn vtb_signal_memory(level: u8) callconv(.c) void {
    const lv: vantablack.PressureLevel = switch (level) {
        0 => .normal,
        1 => .warning,
        else => .critical,
    };
    g_hub.signalMemory(lv);
}

/// `state`: 0 = nominal, 1 = fair, 2 = serious, 3 = critical. Out-of-
/// range clamps to critical. Maps directly from `ProcessInfo
/// .ThermalState` on iOS.
export fn vtb_signal_thermal(state: u8) callconv(.c) void {
    const st: vantablack.ThermalState = switch (state) {
        0 => .nominal,
        1 => .fair,
        2 => .serious,
        else => .critical,
    };
    g_hub.signalThermal(st);
}

// ============================================================================
// Version / capability introspection
// ============================================================================

export fn vtb_version_string() callconv(.c) [*:0]const u8 {
    return "vantablack 0.1.0 (c-abi)";
}

export fn vtb_has_metal() callconv(.c) u8 {
    return @intFromBool(vantablack.metal.metal_enabled);
}

// ============================================================================
// Internal helpers
// ============================================================================

fn castModel(m: ?*VtbModel) ?*ModelHandle {
    const ptr = m orelse return null;
    return @ptrCast(@alignCast(ptr));
}

fn castState(s: ?*VtbState) ?*StateHandle {
    const ptr = s orelse return null;
    return @ptrCast(@alignCast(ptr));
}

// -- tests ----------------------------------------------------------------

test "vtb_signal_memory + vtb_signal_thermal accept all levels without crash" {
    vtb_signal_memory(0);
    vtb_signal_memory(1);
    vtb_signal_memory(2);
    vtb_signal_memory(99);
    vtb_signal_thermal(0);
    vtb_signal_thermal(3);
    vtb_signal_thermal(99);
}

test "vtb_version_string returns a c-string" {
    const v = vtb_version_string();
    const slice = std.mem.span(v);
    try std.testing.expect(slice.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, slice, "vantablack") != null);
}

test "vtb_has_metal returns 0 or 1" {
    const h = vtb_has_metal();
    try std.testing.expect(h == 0 or h == 1);
}

test "vtb_model_open returns null on missing file" {
    const result = vtb_model_open("/nonexistent/vantablack/test/path.gguf", 0);
    try std.testing.expect(result == null);
}
