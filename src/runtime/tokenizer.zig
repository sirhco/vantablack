//! Minimal SentencePiece-style tokenizer decode for Llama GGUF models.
//!
//! Reads `tokenizer.ggml.tokens` (array of strings) from the catalog metadata,
//! materializes a vocab table, and exposes `decode(token_id)` returning the
//! piece bytes.
//!
//! Encoding (string → token IDs) is left to a future revision — the CLI in
//! Task 2 takes pre-tokenized integer IDs.

const std = @import("std");
const Allocator = std.mem.Allocator;

const parser = @import("../core/parser.zig");

pub const TokenizerError = error{
    MissingVocab,
    InvalidVocabPayload,
} || Allocator.Error;

pub const Tokenizer = struct {
    /// Slice of pieces, one per token id. Each piece is a slice into the mmap
    /// region (zero-copy).
    pieces: [][]const u8,
    bos: u32,
    eos: u32,

    pub fn init(allocator: Allocator, catalog: parser.Catalog) TokenizerError!Tokenizer {
        const tokens_v = findValue(catalog, "tokenizer.ggml.tokens") orelse
            return error.MissingVocab;
        const arr = switch (tokens_v) {
            .array => |a| a,
            else => return error.MissingVocab,
        };
        if (arr.elem_type != .string) return error.InvalidVocabPayload;

        const count: usize = std.math.cast(usize, arr.count) orelse return error.InvalidVocabPayload;
        const pieces = try allocator.alloc([]const u8, count);
        errdefer allocator.free(pieces);

        var off: usize = 0;
        for (pieces) |*p| {
            if (off + 8 > arr.raw.len) return error.InvalidVocabPayload;
            const len = std.mem.readInt(u64, arr.raw[off..][0..8], .little);
            off += 8;
            const len_us: usize = std.math.cast(usize, len) orelse
                return error.InvalidVocabPayload;
            if (off + len_us > arr.raw.len) return error.InvalidVocabPayload;
            p.* = arr.raw[off..][0..len_us];
            off += len_us;
        }

        const bos = readU32(catalog, "tokenizer.ggml.bos_token_id") orelse 1;
        const eos = readU32(catalog, "tokenizer.ggml.eos_token_id") orelse 2;

        return .{ .pieces = pieces, .bos = bos, .eos = eos };
    }

    pub fn deinit(self: *Tokenizer, allocator: Allocator) void {
        allocator.free(self.pieces);
        self.* = undefined;
    }

    /// Write the decoded bytes for `token_id` into `writer`. Handles:
    ///  * SentencePiece `▁` (U+2581 = bytes E2 96 81) → space
    ///  * Byte-fallback pieces of the form `<0xHH>` → raw byte 0xHH
    pub fn decodeTo(self: *const Tokenizer, writer: *std.Io.Writer, token_id: u32) !void {
        if (token_id >= self.pieces.len) return;
        const piece = self.pieces[token_id];
        if (piece.len == 6 and piece[0] == '<' and piece[1] == '0' and piece[2] == 'x' and piece[5] == '>') {
            const hi = hexNibble(piece[3]) orelse return writer.writeAll(piece);
            const lo = hexNibble(piece[4]) orelse return writer.writeAll(piece);
            try writer.writeByte(@intCast((hi << 4) | lo));
            return;
        }
        // Replace SentencePiece word-start marker with a real space.
        var i: usize = 0;
        while (i < piece.len) {
            if (i + 3 <= piece.len and piece[i] == 0xE2 and piece[i + 1] == 0x96 and piece[i + 2] == 0x81) {
                try writer.writeByte(' ');
                i += 3;
            } else {
                try writer.writeByte(piece[i]);
                i += 1;
            }
        }
    }
};

fn hexNibble(b: u8) ?u8 {
    return switch (b) {
        '0'...'9' => b - '0',
        'a'...'f' => 10 + (b - 'a'),
        'A'...'F' => 10 + (b - 'A'),
        else => null,
    };
}

fn findValue(catalog: parser.Catalog, key: []const u8) ?parser.MetaValue {
    for (catalog.metadata) |kv| {
        if (std.mem.eql(u8, kv.key, key)) return kv.value;
    }
    return null;
}

fn readU32(catalog: parser.Catalog, key: []const u8) ?u32 {
    const v = findValue(catalog, key) orelse return null;
    return switch (v) {
        .u32 => |x| x,
        .u64 => |x| @intCast(x),
        .i32 => |x| @intCast(x),
        else => null,
    };
}

test "decode byte-fallback piece" {
    var buf: [16]u8 = undefined;
    var fbw: std.Io.Writer = .fixed(&buf);
    const t: Tokenizer = .{
        .pieces = @constCast(&[_][]const u8{ "<0xAB>", "ok" }),
        .bos = 0,
        .eos = 0,
    };
    try t.decodeTo(&fbw, 0);
    try std.testing.expectEqualSlices(u8, &.{0xAB}, fbw.buffered());
}

test "decode SentencePiece space marker" {
    var buf: [16]u8 = undefined;
    var fbw: std.Io.Writer = .fixed(&buf);
    const piece = [_]u8{ 0xE2, 0x96, 0x81, 'h', 'i' };
    const t: Tokenizer = .{
        .pieces = @constCast(&[_][]const u8{piece[0..]}),
        .bos = 0,
        .eos = 0,
    };
    try t.decodeTo(&fbw, 0);
    try std.testing.expectEqualSlices(u8, " hi", fbw.buffered());
}
