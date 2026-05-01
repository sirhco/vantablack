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
    PieceTooLong,
    InvalidUtf8,
} || Allocator.Error;

pub const PieceMap = std.StringHashMapUnmanaged(IdScore);

pub const IdScore = struct { id: u32, score: f32 };

pub const Tokenizer = struct {
    /// Slice of pieces, one per token id. Each piece is a slice into the mmap
    /// region (zero-copy).
    pieces: [][]const u8,
    /// SentencePiece scores per piece (parallel to `pieces`). Higher = preferred merge.
    scores: []f32,
    /// O(1) lookup from piece bytes → token id + score.
    by_piece: PieceMap,
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

        // Scores: optional. Default to 0 if missing.
        const scores = try allocator.alloc(f32, count);
        errdefer allocator.free(scores);
        @memset(scores, 0);
        if (findValue(catalog, "tokenizer.ggml.scores")) |sv| {
            switch (sv) {
                .array => |sa| {
                    if (sa.elem_type == .f32 and sa.count == arr.count) {
                        for (scores, 0..) |*s, i| {
                            const sb = sa.raw[i * 4 ..][0..4];
                            const bits = std.mem.readInt(u32, sb, .little);
                            s.* = @bitCast(bits);
                        }
                    }
                },
                else => {},
            }
        }

        var by_piece: PieceMap = .empty;
        errdefer by_piece.deinit(allocator);
        try by_piece.ensureTotalCapacity(allocator, @intCast(count));
        for (pieces, scores, 0..) |piece, score, i| {
            // Last write wins on duplicates (some llama vocabs have a few).
            try by_piece.put(allocator, piece, .{ .id = @intCast(i), .score = score });
        }

        const bos = readU32(catalog, "tokenizer.ggml.bos_token_id") orelse 1;
        const eos = readU32(catalog, "tokenizer.ggml.eos_token_id") orelse 2;

        return .{
            .pieces = pieces,
            .scores = scores,
            .by_piece = by_piece,
            .bos = bos,
            .eos = eos,
        };
    }

    pub fn deinit(self: *Tokenizer, allocator: Allocator) void {
        self.by_piece.deinit(allocator);
        allocator.free(self.scores);
        allocator.free(self.pieces);
        self.* = undefined;
    }

    /// SentencePiece BPE encoding (matches llama.cpp `llm_tokenizer_spm`).
    /// Spaces are replaced by U+2581 (▁), and a leading ▁ is prepended.
    /// Multi-byte UTF-8 stays intact as initial pieces.
    /// Unknown bytes after merging fall back to `<0xHH>` byte tokens.
    /// Caller owns the returned slice.
    pub fn encode(
        self: *const Tokenizer,
        allocator: Allocator,
        text: []const u8,
        prepend_bos: bool,
    ) TokenizerError![]u32 {
        // Build the canonical input: replace ' ' → "▁", prepend leading "▁".
        var prepared: std.ArrayList(u8) = .empty;
        defer prepared.deinit(allocator);
        try prepared.appendSlice(allocator, "\xe2\x96\x81"); // leading ▁
        for (text) |b| {
            if (b == ' ') try prepared.appendSlice(allocator, "\xe2\x96\x81") else try prepared.append(allocator, b);
        }

        // Initial symbols: one per UTF-8 codepoint (or byte if invalid).
        var symbols: std.ArrayList(Symbol) = .empty;
        defer symbols.deinit(allocator);
        var i: usize = 0;
        while (i < prepared.items.len) {
            const len = utf8Len(prepared.items[i]);
            const take = @min(len, prepared.items.len - i);
            try symbols.append(allocator, .{
                .text = prepared.items[i .. i + take],
                .prev = @intCast(@as(isize, @intCast(symbols.items.len)) - 1),
                .next = -1, // patched below
                .alive = true,
            });
            i += take;
        }
        // Patch next pointers.
        for (symbols.items, 0..) |*sym, idx| {
            sym.next = if (idx + 1 < symbols.items.len) @intCast(idx + 1) else -1;
        }

        // Seed merge queue with all adjacent pairs.
        var queue: BigramQueue = .empty;
        defer queue.deinit(allocator);
        for (1..symbols.items.len) |idx| {
            try self.tryAddBigram(allocator, &queue, symbols.items, @intCast(idx - 1), @intCast(idx));
        }

        // Repeatedly pop best bigram and merge.
        while (queue.pop()) |bg| {
            const left_idx: usize = @intCast(bg.left);
            const right_idx: usize = @intCast(bg.right);
            const left = &symbols.items[left_idx];
            const right = &symbols.items[right_idx];
            if (!left.alive or !right.alive) continue;
            // Stale bigram: text spans changed since enqueue.
            if (left.text.len + right.text.len != bg.merged_len) continue;

            // Merge: extend left to cover both, drop right.
            left.text = prepared.items[charOff(prepared.items, left.text) .. charOff(prepared.items, right.text) + right.text.len];
            right.alive = false;
            left.next = right.next;
            if (right.next >= 0) {
                symbols.items[@intCast(right.next)].prev = bg.left;
            }
            // Try new neighbors.
            try self.tryAddBigram(allocator, &queue, symbols.items, left.prev, bg.left);
            try self.tryAddBigram(allocator, &queue, symbols.items, bg.left, left.next);
        }

        // Emit tokens, with byte-fallback for unknowns.
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(allocator);
        if (prepend_bos) try out.append(allocator, self.bos);

        var cur: i32 = if (symbols.items.len == 0) -1 else 0;
        while (cur >= 0) {
            const sym = symbols.items[@intCast(cur)];
            if (sym.alive) {
                if (self.by_piece.get(sym.text)) |hit| {
                    try out.append(allocator, hit.id);
                } else {
                    // Byte-fallback per byte.
                    for (sym.text) |b| {
                        var buf: [6]u8 = undefined;
                        const name = std.fmt.bufPrint(&buf, "<0x{X:0>2}>", .{b}) catch unreachable;
                        if (self.by_piece.get(name)) |hit| {
                            try out.append(allocator, hit.id);
                        } else {
                            // Last resort: drop. (Should be unreachable for sane vocabs.)
                        }
                    }
                }
            }
            cur = sym.next;
        }
        return try out.toOwnedSlice(allocator);
    }

    fn tryAddBigram(
        self: *const Tokenizer,
        allocator: Allocator,
        queue: *BigramQueue,
        symbols: []const Symbol,
        left: i32,
        right: i32,
    ) !void {
        if (left < 0 or right < 0) return;
        const l = symbols[@intCast(left)];
        const r = symbols[@intCast(right)];
        if (!l.alive or !r.alive) return;
        // Need contiguous bytes to look up the merged piece.
        const start = @intFromPtr(l.text.ptr);
        const end = @intFromPtr(r.text.ptr) + r.text.len;
        const merged_len = end - start;
        if (merged_len > 256) return; // skip absurdly long candidates
        const merged: []const u8 = @as([*]const u8, @ptrFromInt(start))[0..merged_len];
        const hit = self.by_piece.get(merged) orelse return;
        try queue.push(allocator, .{
            .left = left,
            .right = right,
            .score = hit.score,
            .merged_len = merged_len,
        });
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

const Symbol = struct {
    text: []const u8,
    prev: i32,
    next: i32,
    alive: bool,
};

const Bigram = struct {
    left: i32,
    right: i32,
    score: f32,
    merged_len: usize,
};

const BigramQueue = std.PriorityQueue(Bigram, void, bigramCompare);

fn bigramCompare(_: void, a: Bigram, b: Bigram) std.math.Order {
    // Higher score = higher priority. Tie-break: earlier left wins (smaller idx).
    if (a.score > b.score) return .lt;
    if (a.score < b.score) return .gt;
    if (a.left < b.left) return .lt;
    if (a.left > b.left) return .gt;
    return .eq;
}

fn utf8Len(first: u8) usize {
    if ((first & 0x80) == 0) return 1;
    if ((first & 0xE0) == 0xC0) return 2;
    if ((first & 0xF0) == 0xE0) return 3;
    if ((first & 0xF8) == 0xF0) return 4;
    return 1; // invalid leading byte: treat as one-byte symbol
}

fn charOff(buf: []const u8, slice: []const u8) usize {
    return @intFromPtr(slice.ptr) - @intFromPtr(buf.ptr);
}

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
    var pieces = [_][]const u8{ "<0xAB>", "ok" };
    var scores = [_]f32{ 0, 0 };
    const t: Tokenizer = .{
        .pieces = pieces[0..],
        .scores = scores[0..],
        .by_piece = .empty,
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
    var pieces = [_][]const u8{piece[0..]};
    var scores = [_]f32{0};
    const t: Tokenizer = .{
        .pieces = pieces[0..],
        .scores = scores[0..],
        .by_piece = .empty,
        .bos = 0,
        .eos = 0,
    };
    try t.decodeTo(&fbw, 0);
    try std.testing.expectEqualSlices(u8, " hi", fbw.buffered());
}
