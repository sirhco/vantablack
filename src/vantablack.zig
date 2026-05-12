//! Public module index for the `vantablack` package.

pub const mapper = @import("core/mapper.zig");
pub const parser = @import("core/parser.zig");
pub const safetensors = @import("core/safetensors.zig");
pub const hf_config = @import("core/hf_config.zig");
pub const hf_loader = @import("core/hf_loader.zig");
pub const kernels = @import("kernels/comptime_gen.zig");
pub const simd = @import("kernels/simd.zig");
pub const math = @import("kernels/math.zig");
pub const mlx = @import("kernels/mlx.zig");
pub const kv_cache = @import("runtime/kv_cache.zig");
pub const model = @import("runtime/model.zig");
pub const forward = @import("runtime/forward.zig");
pub const prefill = @import("runtime/prefill.zig");
pub const tokenizer = @import("runtime/tokenizer.zig");
pub const byte_level_bpe = @import("runtime/byte_level_bpe.zig");
pub const unicode_table = @import("runtime/unicode_table.zig");
pub const sampler = @import("runtime/sampler.zig");
pub const pool = @import("runtime/pool.zig");
pub const chat_template = @import("runtime/chat_template.zig");
pub const server = @import("runtime/server.zig");
pub const metal = @import("metal/bridge.zig");
pub const metal_backend = @import("runtime/metal_backend.zig");
pub const backend = @import("runtime/backend.zig");
pub const backend_cpu = @import("runtime/backend_cpu.zig");
pub const backend_metal = @import("runtime/backend_metal.zig");
pub const stream = @import("runtime/stream.zig");
pub const generateStream = stream.generateStream;
pub const TokenCallback = stream.TokenCallback;
pub const pressure = @import("runtime/pressure.zig");
pub const PressureHub = pressure.Hub;
pub const PressureSink = pressure.Sink;

pub const Backend = backend.Backend;
pub const BackendVTable = backend.VTable;
pub const BackendCapabilities = backend.Capabilities;
pub const PressureLevel = backend.PressureLevel;
pub const ThermalState = backend.ThermalState;
pub const CpuBackend = backend_cpu.CpuBackend;
pub const MetalDispatch = backend_metal.MetalDispatch;

pub const ThreadPool = pool.ThreadPool;
pub const Server = server.Server;
pub const MetalBackend = metal_backend.MetalBackend;

pub const Tokenizer = tokenizer.Tokenizer;
pub const Sampler = sampler.Sampler;
pub const SamplerConfig = sampler.Config;

pub const ModelMapper = mapper.ModelMapper;
pub const Catalog = parser.Catalog;
pub const TensorDesc = parser.TensorDesc;
pub const GgmlType = parser.GgmlType;
pub const MetaValue = parser.MetaValue;
pub const MetaKv = parser.MetaKv;

pub const QuantType = kernels.QuantType;
pub const Kernel = kernels.Kernel;
pub const dispatch = kernels.dispatch;

pub const KvCache = kv_cache.KvCache;
pub const LlamaConfig = model.LlamaConfig;
pub const Model = model.Model;
pub const State = forward.State;
pub const step = forward.step;

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
