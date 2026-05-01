//! HuggingFace `config.json` → `LlamaConfig` adapter.
//!
//! MLX-released Llama-architecture models ship the same `config.json` shape
//! as the original HuggingFace transformers checkpoint, plus an optional
//! `quantization` block that records `bits` and `group_size`. This module
//! parses that JSON into a struct that mirrors what `runtime/model.zig`
//! produces from GGUF metadata, so the downstream forward pass stays
//! quant-format-agnostic.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const ParseError = error{
    BadJson,
    MissingField,
    UnsupportedArchitecture,
    OutOfMemory,
};

/// Subset of HuggingFace `config.json` we care about. Field names mirror the
/// upstream JSON keys so the mapping stays obvious.
pub const HfConfig = struct {
    model_type: []const u8,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    vocab_size: usize,
    rope_theta: f32 = 10000.0,
    rms_norm_eps: f32 = 1e-5,
    /// Some configs split head_dim out from hidden_size/num_attention_heads
    /// (rare for Llama; standard for Qwen). `null` ⇒ derive as
    /// hidden_size / num_attention_heads.
    head_dim: ?usize = null,
    /// `lm_head.weight` tied to `model.embed_tokens.weight` — common for
    /// small Llama variants (TinyLlama, Llama-3.2-1B).
    tie_word_embeddings: bool = false,
    /// Present when the checkpoint was produced via MLX's `quantize`. Drives
    /// per-layer tensor naming + matmul kernel selection.
    quantization: ?Quantization = null,
};

pub const Quantization = struct {
    /// Bits per quantized weight. MLX supports 2/3/4/6/8.
    bits: u32,
    /// Number of weight elements that share one (scale, bias) pair.
    /// Standard MLX defaults are 64 or 128.
    group_size: u32,
};

/// Parse `config.json` bytes. All allocations land in `arena`.
pub fn parse(arena: Allocator, json_bytes: []const u8) ParseError!HfConfig {
    const parsed = std.json.parseFromSlice(std.json.Value, arena, json_bytes, .{}) catch return error.BadJson;
    const root = parsed.value;
    if (root != .object) return error.BadJson;

    const obj = root.object;

    const model_type = try strField(obj, "model_type");
    // Llama-arch families share tensor naming. MLX also publishes Mistral,
    // Qwen2, Phi3 etc. with the same general shape; gate to llama for now.
    if (!std.mem.eql(u8, model_type, "llama")) return error.UnsupportedArchitecture;

    var cfg = HfConfig{
        .model_type = try arena.dupe(u8, model_type),
        .hidden_size = try usizeField(obj, "hidden_size"),
        .num_hidden_layers = try usizeField(obj, "num_hidden_layers"),
        .num_attention_heads = try usizeField(obj, "num_attention_heads"),
        .num_key_value_heads = optUsizeField(obj, "num_key_value_heads") orelse
            try usizeField(obj, "num_attention_heads"),
        .intermediate_size = try usizeField(obj, "intermediate_size"),
        .max_position_embeddings = try usizeField(obj, "max_position_embeddings"),
        .vocab_size = try usizeField(obj, "vocab_size"),
    };
    if (optF32Field(obj, "rope_theta")) |x| cfg.rope_theta = x;
    if (optF32Field(obj, "rms_norm_eps")) |x| cfg.rms_norm_eps = x;
    if (optUsizeField(obj, "head_dim")) |x| cfg.head_dim = x;
    if (obj.get("tie_word_embeddings")) |v| {
        if (v == .bool) cfg.tie_word_embeddings = v.bool;
    }

    if (obj.get("quantization")) |qv| {
        if (qv == .object) {
            const bits = (optUsizeField(qv.object, "bits") orelse 0);
            const grp = (optUsizeField(qv.object, "group_size") orelse 0);
            if (bits != 0 and grp != 0) {
                cfg.quantization = .{
                    .bits = @intCast(bits),
                    .group_size = @intCast(grp),
                };
            }
        }
    }

    return cfg;
}

fn strField(obj: std.json.ObjectMap, key: []const u8) ParseError![]const u8 {
    const v = obj.get(key) orelse return error.MissingField;
    if (v != .string) return error.BadJson;
    return v.string;
}

fn usizeField(obj: std.json.ObjectMap, key: []const u8) ParseError!usize {
    const v = obj.get(key) orelse return error.MissingField;
    return switch (v) {
        .integer => |x| if (x < 0) error.BadJson else @intCast(x),
        else => error.BadJson,
    };
}

fn optUsizeField(obj: std.json.ObjectMap, key: []const u8) ?usize {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .integer => |x| if (x < 0) null else @intCast(x),
        else => null,
    };
}

fn optF32Field(obj: std.json.ObjectMap, key: []const u8) ?f32 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .float => |x| @floatCast(x),
        .integer => |x| @floatFromInt(x),
        else => null,
    };
}

// -- tests ----------------------------------------------------------------

test "parse: TinyLlama-Chat config (FP16, no quantization block)" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();

    const json =
        \\{
        \\  "model_type": "llama",
        \\  "hidden_size": 2048,
        \\  "num_hidden_layers": 22,
        \\  "num_attention_heads": 32,
        \\  "num_key_value_heads": 4,
        \\  "intermediate_size": 5632,
        \\  "max_position_embeddings": 2048,
        \\  "vocab_size": 32000,
        \\  "rope_theta": 10000.0,
        \\  "rms_norm_eps": 1e-5,
        \\  "tie_word_embeddings": false
        \\}
    ;
    const cfg = try parse(arena.allocator(), json);
    try std.testing.expectEqualStrings("llama", cfg.model_type);
    try std.testing.expectEqual(@as(usize, 2048), cfg.hidden_size);
    try std.testing.expectEqual(@as(usize, 22), cfg.num_hidden_layers);
    try std.testing.expectEqual(@as(usize, 4), cfg.num_key_value_heads);
    try std.testing.expectEqual(@as(usize, 5632), cfg.intermediate_size);
    try std.testing.expectEqual(@as(f32, 1e-5), cfg.rms_norm_eps);
    try std.testing.expect(cfg.quantization == null);
    try std.testing.expectEqual(false, cfg.tie_word_embeddings);
}

test "parse: MLX-quantized 4-bit config" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();

    const json =
        \\{
        \\  "model_type": "llama",
        \\  "hidden_size": 2048,
        \\  "num_hidden_layers": 22,
        \\  "num_attention_heads": 32,
        \\  "num_key_value_heads": 4,
        \\  "intermediate_size": 5632,
        \\  "max_position_embeddings": 2048,
        \\  "vocab_size": 32000,
        \\  "quantization": {"bits": 4, "group_size": 64}
        \\}
    ;
    const cfg = try parse(arena.allocator(), json);
    try std.testing.expect(cfg.quantization != null);
    try std.testing.expectEqual(@as(u32, 4), cfg.quantization.?.bits);
    try std.testing.expectEqual(@as(u32, 64), cfg.quantization.?.group_size);
}

test "parse: rejects non-llama model_type" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();

    const json =
        \\{"model_type": "bert", "hidden_size": 1, "num_hidden_layers": 1,
        \\ "num_attention_heads": 1, "intermediate_size": 1,
        \\ "max_position_embeddings": 1, "vocab_size": 1}
    ;
    try std.testing.expectError(error.UnsupportedArchitecture, parse(arena.allocator(), json));
}
