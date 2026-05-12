//! CPU `Backend` implementation.
//!
//! Wraps the existing per-op CPU path (`forward.cpuLayerStep` +
//! `rmsNormTyped` + `matmulRuntime`) behind a stable vtable. Optionally
//! holds a `*MetalBackend` pointer for the historical "hybrid" path where
//! the model overall is not GPU-eligible but individual Q8_0 projections
//! are offloaded inside `matmulRuntime`. That mode preserves bit-equality
//! with the pre-vtable build for mixed-quant models.

const std = @import("std");

const backend_mod = @import("backend.zig");
const forward_mod = @import("forward.zig");
const kv_cache_mod = @import("kv_cache.zig");
const model_mod = @import("model.zig");
const pool_mod = @import("pool.zig");
const metal_backend_mod = @import("metal_backend.zig");

const Backend = backend_mod.Backend;
const Capabilities = backend_mod.Capabilities;
const PressureLevel = backend_mod.PressureLevel;
const ThermalState = backend_mod.ThermalState;
const VTable = backend_mod.VTable;
const StatePtr = backend_mod.StatePtr;

const State = forward_mod.State;
const KvCache = kv_cache_mod.KvCache;
const Model = model_mod.Model;
const LayerWeights = model_mod.LayerWeights;
const LlamaConfig = model_mod.LlamaConfig;
const ThreadPool = pool_mod.ThreadPool;
const MetalBackend = metal_backend_mod.MetalBackend;

pub const CpuBackend = struct {
    pool: *ThreadPool,
    /// Optional pointer to a Metal backend for per-projection Q8_0/K-quant
    /// offload inside `matmulRuntime`. Null on iOS / non-Apple targets.
    /// This path is only exercised when the model as a whole is *not* GPU-
    /// eligible (e.g. mixed-quant), so the orchestrator stays on the CPU
    /// vtable but individual matmuls can still hit the GPU.
    metal_fallback: ?*MetalBackend = null,

    pub fn backend(self: *CpuBackend) Backend {
        return .{
            .ptr = self,
            .vtable = &vtable,
        };
    }
};

const vtable: VTable = .{
    .layer_step = layerStep,
    .final_step = finalStep,
    .finalize_token = finalizeToken,
    .capabilities = capabilities,
    .on_pressure = onPressure,
    .on_thermal = onThermal,
};

fn castSelf(ctx: *anyopaque) *CpuBackend {
    return @ptrCast(@alignCast(ctx));
}

fn castState(state: StatePtr) *State {
    return @ptrCast(@alignCast(state));
}

fn layerStep(
    ctx: *anyopaque,
    layer: *const LayerWeights,
    cache: *KvCache,
    state: StatePtr,
    c: LlamaConfig,
    layer_idx: usize,
) anyerror!void {
    const self = castSelf(ctx);
    try forward_mod.cpuLayerStep(
        self.pool,
        self.metal_fallback,
        layer.*,
        castState(state),
        cache,
        c,
        layer_idx,
    );
}

fn finalStep(
    ctx: *anyopaque,
    m: *const Model,
    c: LlamaConfig,
    state: StatePtr,
) anyerror!void {
    const self = castSelf(ctx);
    const s = castState(state);
    try forward_mod.rmsNormTyped(s.x, m.output_norm, c.rms_eps);
    try forward_mod.matmulRuntime(
        self.pool,
        self.metal_fallback,
        s.logits,
        m.output_w,
        s.x,
        c.vocab_size,
        c.dim,
    );
}

fn finalizeToken(_: *anyopaque) anyerror!void {
    // CPU path is synchronous — every per-op call already returned with its
    // result visible to the host. No barrier needed.
}

fn capabilities(ctx: *anyopaque) Capabilities {
    const self = castSelf(ctx);
    return .{
        .supports_fused_qkv = false,
        .supports_kquant_gpu = self.metal_fallback != null,
        .supports_mlx_q4 = false,
        .unified_memory = true,
        .name = "cpu",
    };
}

fn onPressure(ctx: *anyopaque, level: PressureLevel) void {
    const self = castSelf(ctx);
    // Thermal-style throttling — under memory pressure we don't need fewer
    // workers, but reducing parallelism does reduce concurrent allocator
    // pressure when scratch buffers are paged.
    switch (level) {
        .normal => {},
        .warning => self.pool.setActiveWorkersClamped(self.pool.workerCount() / 2),
        .critical => self.pool.setActiveWorkersClamped(1),
    }
}

fn onThermal(ctx: *anyopaque, state: ThermalState) void {
    const self = castSelf(ctx);
    switch (state) {
        .nominal => self.pool.setActiveWorkersClamped(self.pool.workerCount()),
        .fair => self.pool.setActiveWorkersClamped(self.pool.workerCount() * 3 / 4),
        .serious => self.pool.setActiveWorkersClamped(self.pool.workerCount() / 2),
        .critical => self.pool.setActiveWorkersClamped(1),
    }
}
