//! Llama-architecture model binding.
//!
//! Reads `llama.*` metadata keys to build a typed config, then resolves the
//! per-layer weight tensors from the GGUF catalog into raw byte slices that
//! point directly into the mmap'd region. No weight bytes are copied.

const std = @import("std");
const Allocator = std.mem.Allocator;

const parser = @import("../core/parser.zig");
const mapper_mod = @import("../core/mapper.zig");
const ModelMapper = mapper_mod.ModelMapper;

pub const ConfigError = error{
    MissingMetadata,
    UnsupportedArchitecture,
    UnsupportedWeightType,
};

pub const ModelError = error{
    MissingTensor,
    UnexpectedTensorShape,
} || ConfigError || mapper_mod.TensorSliceError || Allocator.Error;

pub const LlamaConfig = struct {
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ffn_dim: usize,
    vocab_size: usize,
    max_seq: usize,
    rope_base: f32 = 10000.0,
    rms_eps: f32 = 1e-5,

    pub fn fromCatalog(catalog: parser.Catalog) ConfigError!LlamaConfig {
        const arch_v = findValue(catalog, "general.architecture") orelse return error.MissingMetadata;
        const arch = switch (arch_v) {
            .string => |s| s,
            else => return error.MissingMetadata,
        };
        if (!std.mem.eql(u8, arch, "llama")) return error.UnsupportedArchitecture;

        const dim = try metaUsize(catalog, "llama.embedding_length");
        const n_layers = try metaUsize(catalog, "llama.block_count");
        const n_heads = try metaUsize(catalog, "llama.attention.head_count");
        const ffn_dim = try metaUsize(catalog, "llama.feed_forward_length");
        const max_seq = try metaUsize(catalog, "llama.context_length");
        const n_kv_heads = metaUsize(catalog, "llama.attention.head_count_kv") catch n_heads;
        const rope_base = metaF32(catalog, "llama.rope.freq_base") catch 10000.0;
        const rms_eps = metaF32(catalog, "llama.attention.layer_norm_rms_epsilon") catch 1e-5;

        // Vocab size: prefer llama.vocab_size; fall back to tokenizer array len.
        const vocab_size = blk: {
            if (metaUsize(catalog, "llama.vocab_size")) |v| break :blk v else |_| {}
            if (findValue(catalog, "tokenizer.ggml.tokens")) |v| switch (v) {
                .array => |arr| break :blk @as(usize, @intCast(arr.count)),
                else => {},
            };
            return error.MissingMetadata;
        };

        if (n_heads == 0) return error.MissingMetadata;
        const head_dim = dim / n_heads;

        return .{
            .dim = dim,
            .n_layers = n_layers,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .ffn_dim = ffn_dim,
            .vocab_size = vocab_size,
            .max_seq = max_seq,
            .rope_base = rope_base,
            .rms_eps = rms_eps,
        };
    }
};

pub const TypedTensor = struct {
    bytes: []const u8,
    ggml_type: parser.GgmlType,
};

pub const LayerWeights = struct {
    attn_norm: TypedTensor,
    attn_q: TypedTensor,
    attn_k: TypedTensor,
    attn_v: TypedTensor,
    attn_o: TypedTensor,
    ffn_norm: TypedTensor,
    ffn_gate: TypedTensor,
    ffn_up: TypedTensor,
    ffn_down: TypedTensor,
};

pub const Model = struct {
    config: LlamaConfig,
    token_embd: TypedTensor,
    output_norm: TypedTensor,
    output_w: TypedTensor,
    layers: []LayerWeights,

    pub fn init(allocator: Allocator, m: *const ModelMapper) ModelError!Model {
        const config = try LlamaConfig.fromCatalog(m.catalog);

        const layers = try allocator.alloc(LayerWeights, config.n_layers);
        errdefer allocator.free(layers);

        var name_buf: [64]u8 = undefined;
        for (layers, 0..) |*lw, i| {
            lw.* = .{
                .attn_norm = try fetch(m, fmt(&name_buf, "blk.{d}.attn_norm.weight", .{i})),
                .attn_q = try fetch(m, fmt(&name_buf, "blk.{d}.attn_q.weight", .{i})),
                .attn_k = try fetch(m, fmt(&name_buf, "blk.{d}.attn_k.weight", .{i})),
                .attn_v = try fetch(m, fmt(&name_buf, "blk.{d}.attn_v.weight", .{i})),
                .attn_o = try fetch(m, fmt(&name_buf, "blk.{d}.attn_output.weight", .{i})),
                .ffn_norm = try fetch(m, fmt(&name_buf, "blk.{d}.ffn_norm.weight", .{i})),
                .ffn_gate = try fetch(m, fmt(&name_buf, "blk.{d}.ffn_gate.weight", .{i})),
                .ffn_up = try fetch(m, fmt(&name_buf, "blk.{d}.ffn_up.weight", .{i})),
                .ffn_down = try fetch(m, fmt(&name_buf, "blk.{d}.ffn_down.weight", .{i})),
            };
        }

        const token_embd = try fetch(m, "token_embd.weight");
        const output_norm = try fetch(m, "output_norm.weight");
        // Some llamas tie output to token_embd (no `output.weight`); fall back.
        const output_w = fetch(m, "output.weight") catch token_embd;

        return .{
            .config = config,
            .token_embd = token_embd,
            .output_norm = output_norm,
            .output_w = output_w,
            .layers = layers,
        };
    }

    pub fn deinit(self: *Model, allocator: Allocator) void {
        allocator.free(self.layers);
        self.* = undefined;
    }
};

fn fmt(buf: []u8, comptime template: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, template, args) catch unreachable;
}

fn fetch(m: *const ModelMapper, name: []const u8) ModelError!TypedTensor {
    const desc = m.catalog.find(name) orelse return error.MissingTensor;
    const bytes = try m.tensorSliceFromDesc(desc);
    return .{ .bytes = bytes, .ggml_type = desc.ggml_type };
}

fn findValue(catalog: parser.Catalog, key: []const u8) ?parser.MetaValue {
    for (catalog.metadata) |kv| {
        if (std.mem.eql(u8, kv.key, key)) return kv.value;
    }
    return null;
}

fn metaUsize(catalog: parser.Catalog, key: []const u8) ConfigError!usize {
    const v = findValue(catalog, key) orelse return error.MissingMetadata;
    return switch (v) {
        .u8 => |x| x,
        .u16 => |x| x,
        .u32 => |x| x,
        .u64 => |x| @intCast(x),
        .i8 => |x| @intCast(x),
        .i16 => |x| @intCast(x),
        .i32 => |x| @intCast(x),
        .i64 => |x| @intCast(x),
        else => error.MissingMetadata,
    };
}

fn metaF32(catalog: parser.Catalog, key: []const u8) ConfigError!f32 {
    const v = findValue(catalog, key) orelse return error.MissingMetadata;
    return switch (v) {
        .f32 => |x| x,
        .f64 => |x| @floatCast(x),
        else => error.MissingMetadata,
    };
}

/// Map a GGML tensor type into the dispatcher's QuantType taxonomy.
pub fn quantOf(t: parser.GgmlType) ConfigError!@import("../kernels/comptime_gen.zig").QuantType {
    return switch (t) {
        .f32 => .f32,
        .f16 => .f16,
        .q8_0 => .q8_0,
        .q4_k => .q4_k,
        .tq2_0, .tq1_0 => .ternary158,
        else => error.UnsupportedWeightType,
    };
}
