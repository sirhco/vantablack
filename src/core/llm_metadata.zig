//! Minimal protobuf wire-format scanner + `LlmMetadataProto` reader.
//!
//! Decodes proto3 wire format (varint / fixed64 / length-delimited /
//! fixed32). No code generation, no full schema; we read only the
//! fields vantablack needs out of the `LlmMetadata` message:
//!
//!   field 1:  TokenUnion start_token         (length-delimited)
//!   field 2:  repeated TokenUnion stop_tokens (length-delimited)
//!   field 3:  PromptTemplates prompt_templates (length-delimited)
//!   field 5:  int32 max_num_tokens            (varint)
//!   field 6:  LlmModelType llm_model_type     (length-delimited)
//!   field 7:  string jinja_prompt_template    (length-delimited)
//!
//! Unknown fields are skipped. Repeated `stop_tokens` are returned as
//! a slice of length-delimited byte ranges.

const std = @import("std");

pub const WireType = enum(u3) {
    varint = 0,
    fixed64 = 1,
    length_delimited = 2,
    start_group = 3,
    end_group = 4,
    fixed32 = 5,
};

pub const Error = error{
    Truncated,
    InvalidWireType,
    VarintOverflow,
    OutOfMemory,
};

pub const FieldData = union(enum) {
    varint: u64,
    bytes: []const u8,
    fixed64: u64,
    fixed32: u32,
};

pub const Field = struct {
    number: u32,
    data: FieldData,
};

/// Scans a buffer of proto3 wire-format bytes one field at a time.
pub const Iterator = struct {
    bytes: []const u8,
    pos: usize = 0,

    pub fn init(bytes: []const u8) Iterator {
        return .{ .bytes = bytes };
    }

    pub fn next(self: *Iterator) Error!?Field {
        if (self.pos >= self.bytes.len) return null;
        const tag = try self.readVarint();
        const field_no: u32 = @intCast(tag >> 3);
        const wire_n: u3 = @intCast(tag & 0x7);
        const wire: WireType = @enumFromInt(wire_n);
        return switch (wire) {
            .varint => Field{ .number = field_no, .data = .{ .varint = try self.readVarint() } },
            .fixed64 => Field{ .number = field_no, .data = .{ .fixed64 = try self.readFixed64() } },
            .length_delimited => Field{
                .number = field_no,
                .data = .{ .bytes = try self.readLengthDelimited() },
            },
            .fixed32 => Field{ .number = field_no, .data = .{ .fixed32 = try self.readFixed32() } },
            else => error.InvalidWireType, // start/end group not supported in proto3
        };
    }

    fn readVarint(self: *Iterator) Error!u64 {
        var result: u64 = 0;
        var shift: u6 = 0;
        var byte_count: usize = 0;
        while (byte_count < 10) : (byte_count += 1) {
            if (self.pos >= self.bytes.len) return error.Truncated;
            const b = self.bytes[self.pos];
            self.pos += 1;
            result |= @as(u64, b & 0x7f) << shift;
            if (b & 0x80 == 0) return result;
            if (shift > 57) return error.VarintOverflow;
            shift += 7;
        }
        return error.VarintOverflow;
    }

    fn readFixed64(self: *Iterator) Error!u64 {
        if (self.pos + 8 > self.bytes.len) return error.Truncated;
        const v = std.mem.readInt(u64, self.bytes[self.pos..][0..8], .little);
        self.pos += 8;
        return v;
    }

    fn readFixed32(self: *Iterator) Error!u32 {
        if (self.pos + 4 > self.bytes.len) return error.Truncated;
        const v = std.mem.readInt(u32, self.bytes[self.pos..][0..4], .little);
        self.pos += 4;
        return v;
    }

    fn readLengthDelimited(self: *Iterator) Error![]const u8 {
        const len_u64 = try self.readVarint();
        const len: usize = @intCast(len_u64);
        if (self.pos + len > self.bytes.len) return error.Truncated;
        const slice = self.bytes[self.pos..][0..len];
        self.pos += len;
        return slice;
    }
};

/// High-level view over an `LlmMetadata` message. Sub-messages
/// (`TokenUnion`, `PromptTemplates`, `LlmModelType`) are kept as raw
/// byte slices so callers can decode them lazily — we do not parse
/// every layer eagerly, only the scalar fields we need now.
pub const LlmMetadata = struct {
    /// Raw `TokenUnion` bytes (or null if absent). Decode via `TokenUnion.parse`.
    start_token_bytes: ?[]const u8 = null,
    /// Raw `TokenUnion` bytes for each repeated stop token.
    stop_tokens_bytes: [][]const u8 = &.{},
    /// Raw `PromptTemplates` bytes (or null if absent).
    prompt_templates_bytes: ?[]const u8 = null,
    /// Maximum number of tokens (proto field `max_num_tokens`).
    max_num_tokens: i32 = 0,
    /// Raw `LlmModelType` bytes (or null if absent).
    llm_model_type_bytes: ?[]const u8 = null,
    /// Jinja prompt template string (or null).
    jinja_prompt_template: ?[]const u8 = null,

    allocator: std.mem.Allocator,

    pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) Error!LlmMetadata {
        var self: LlmMetadata = .{ .allocator = allocator };
        // First pass: count repeated stop_tokens so we can size the slice.
        var count_iter = Iterator.init(bytes);
        var n_stop: usize = 0;
        while (try count_iter.next()) |f| {
            if (f.number == 2 and f.data == .bytes) n_stop += 1;
        }
        if (n_stop > 0) {
            self.stop_tokens_bytes = try allocator.alloc([]const u8, n_stop);
        }

        // Second pass: populate fields.
        var iter = Iterator.init(bytes);
        var stop_idx: usize = 0;
        while (try iter.next()) |f| {
            switch (f.number) {
                1 => if (f.data == .bytes) {
                    self.start_token_bytes = f.data.bytes;
                },
                2 => if (f.data == .bytes) {
                    self.stop_tokens_bytes[stop_idx] = f.data.bytes;
                    stop_idx += 1;
                },
                3 => if (f.data == .bytes) {
                    self.prompt_templates_bytes = f.data.bytes;
                },
                5 => if (f.data == .varint) {
                    // proto3 int32 is encoded as varint, but the in-memory
                    // value is signed. ZigZag is only for sint32/sint64;
                    // a negative int32 is sign-extended to 10-byte varint.
                    self.max_num_tokens = @bitCast(@as(u32, @truncate(f.data.varint)));
                },
                6 => if (f.data == .bytes) {
                    self.llm_model_type_bytes = f.data.bytes;
                },
                7 => if (f.data == .bytes) {
                    self.jinja_prompt_template = f.data.bytes;
                },
                else => {},
            }
        }
        return self;
    }

    pub fn deinit(self: *LlmMetadata) void {
        if (self.stop_tokens_bytes.len > 0) self.allocator.free(self.stop_tokens_bytes);
        self.* = undefined;
    }
};

/// Resolved model-type tag. The `LlmModelType` proto is a `oneof` over
/// concrete model messages — we return which one was selected so the
/// caller can choose the right Llama / Gemma forward path.
pub const ModelTypeTag = enum(u32) {
    generic_model = 1,
    gemma3n = 2,
    function_gemma = 3,
    gemma3 = 4,
    qwen3 = 5,
    qwen2p5 = 7,
    gemma4 = 8,
    fast_vlm = 9,
    _,

    pub fn name(self: ModelTypeTag) []const u8 {
        return switch (self) {
            .generic_model => "GenericModel",
            .gemma3n => "Gemma3N",
            .function_gemma => "FunctionGemma",
            .gemma3 => "Gemma3",
            .qwen3 => "Qwen3",
            .qwen2p5 => "Qwen2p5",
            .gemma4 => "Gemma4",
            .fast_vlm => "FastVlm",
            _ => "Unknown",
        };
    }
};

/// Decode the first present `oneof` arm in an `LlmModelType` message
/// and return its tag. Returns null if no arm is set.
pub fn detectModelType(bytes: []const u8) Error!?ModelTypeTag {
    var iter = Iterator.init(bytes);
    while (try iter.next()) |f| {
        // First length-delimited field with a known number wins.
        if (f.data == .bytes) {
            return @enumFromInt(f.number);
        }
    }
    return null;
}

// -- tests ----------------------------------------------------------------

test "Iterator decodes a varint field" {
    // Field 5 (max_num_tokens), wire varint = 0:
    //   tag = (5 << 3) | 0 = 40 = 0x28
    //   value = 8192 = 0x2000 (14 bits): low 7 = 0, high 7 = 0x40
    //     → varint bytes 0x80 0x40
    var bytes = [_]u8{ 0x28, 0x80, 0x40 };
    var iter = Iterator.init(&bytes);
    const f = (try iter.next()).?;
    try std.testing.expectEqual(@as(u32, 5), f.number);
    try std.testing.expectEqual(@as(u64, 8192), f.data.varint);
    try std.testing.expect((try iter.next()) == null);
}

test "Iterator decodes length-delimited bytes" {
    // Field 7 (jinja_prompt_template), wire 2:
    //   tag = (7 << 3) | 2 = 58 = 0x3A
    //   len = 5 (varint single byte)
    //   bytes = "hello"
    var bytes = [_]u8{ 0x3A, 0x05, 'h', 'e', 'l', 'l', 'o' };
    var iter = Iterator.init(&bytes);
    const f = (try iter.next()).?;
    try std.testing.expectEqual(@as(u32, 7), f.number);
    try std.testing.expectEqualStrings("hello", f.data.bytes);
}

test "LlmMetadata.parse picks max_num_tokens + jinja template" {
    //   field 5 varint 4096   (4096 = 0x1000 → low7=0, high7=0x20 → 0x80 0x20)
    //   field 7 length 4 "xyz!"
    var bytes = [_]u8{
        0x28, 0x80, 0x20,
        0x3A, 0x04, 'x', 'y', 'z', '!',
    };
    var meta = try LlmMetadata.parse(std.testing.allocator, &bytes);
    defer meta.deinit();
    try std.testing.expectEqual(@as(i32, 4096), meta.max_num_tokens);
    try std.testing.expectEqualStrings("xyz!", meta.jinja_prompt_template.?);
}

test "detectModelType picks first oneof arm" {
    // Field 8 (gemma4), length-delimited, len 0:
    //   tag = (8 << 3) | 2 = 0x42, len = 0
    var bytes = [_]u8{ 0x42, 0x00 };
    const tag = (try detectModelType(&bytes)).?;
    try std.testing.expectEqual(ModelTypeTag.gemma4, tag);
    try std.testing.expectEqualStrings("Gemma4", tag.name());
}
