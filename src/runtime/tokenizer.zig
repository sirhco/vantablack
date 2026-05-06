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
const byte_level_bpe = @import("byte_level_bpe.zig");

pub const TokenizerError = error{
    MissingVocab,
    InvalidVocabPayload,
    PieceTooLong,
    InvalidUtf8,
} || Allocator.Error;

pub const PieceMap = std.StringHashMapUnmanaged(IdScore);

pub const IdScore = struct { id: u32, score: f32 };

/// Selects how `Tokenizer.encode` prepares text before greedy BPE merge.
pub const Flavor = enum {
    /// Llama 1/2-style: replace ' ' with U+2581 (▁), prepend leading ▁,
    /// fall back to <0xHH> byte tokens for unknown spans.
    sentencepiece,
    /// GPT-2 / Llama-3-style: pre-tokenizer split on Unicode classes,
    /// each byte mapped through the GPT-2 byte→Unicode alphabet, BPE
    /// merge runs per pre-token. No byte fallback (every mapped byte is
    /// guaranteed to be in vocab).
    byte_level,
};

/// Pre-tokenizer regex flavor for the byte-level path. GPT-2 splits text
/// purely by Unicode category. Llama-3 / cl100k_base adds case-insensitive
/// contractions, leading-non-LN binding before letters, 1-3 digit chunking,
/// and the `\s+(?!\S)` whitespace peel.
pub const ByteSplit = enum { gpt2, llama3 };

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
    flavor: Flavor = .sentencepiece,
    byte_split: ByteSplit = .gpt2,

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

    /// Encode `text` to token IDs. Dispatches to the SentencePiece or
    /// byte-level pipeline based on `self.flavor`.
    pub fn encode(
        self: *const Tokenizer,
        allocator: Allocator,
        text: []const u8,
        prepend_bos: bool,
    ) TokenizerError![]u32 {
        return switch (self.flavor) {
            .sentencepiece => self.encodeSentencePiece(allocator, text, prepend_bos),
            .byte_level => self.encodeByteLevel(allocator, text, prepend_bos),
        };
    }

    /// SentencePiece BPE encoding (matches llama.cpp `llm_tokenizer_spm`).
    /// Spaces are replaced by U+2581 (▁), and a leading ▁ is prepended.
    /// Multi-byte UTF-8 stays intact as initial pieces.
    /// Unknown bytes after merging fall back to `<0xHH>` byte tokens.
    /// Caller owns the returned slice.
    pub fn encodeSentencePiece(
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

    /// Byte-level (tiktoken-style) encoder for GPT-2 / Llama-3 / cl100k
    /// vocabularies. Pre-tokenizes via `byte_level_bpe.splitGpt2`, maps
    /// every byte through the GPT-2 byte→Unicode alphabet, runs greedy
    /// BPE per pre-token. No byte fallback — byte-level vocabs include
    /// every mapped byte by construction.
    pub fn encodeByteLevel(
        self: *const Tokenizer,
        allocator: Allocator,
        text: []const u8,
        prepend_bos: bool,
    ) TokenizerError![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(allocator);
        if (prepend_bos) try out.append(allocator, self.bos);

        var splits: std.ArrayList(byte_level_bpe.Span) = .empty;
        defer splits.deinit(allocator);
        switch (self.byte_split) {
            .gpt2 => byte_level_bpe.splitGpt2(text, &splits, allocator) catch return error.InvalidUtf8,
            .llama3 => byte_level_bpe.splitLlama3(text, &splits, allocator) catch return error.InvalidUtf8,
        }

        var mapped: std.ArrayList(u8) = .empty;
        defer mapped.deinit(allocator);

        for (splits.items) |sp| {
            mapped.clearRetainingCapacity();
            for (text[sp.start..sp.end]) |b| {
                byte_level_bpe.utf8Append(byte_level_bpe.byte_to_unicode[b], &mapped, allocator) catch return error.InvalidUtf8;
            }
            try self.bpeMergeMapped(allocator, mapped.items, &out);
        }
        return out.toOwnedSlice(allocator);
    }

    /// Run greedy BPE on a mapped UTF-8 string and append token IDs to `out`.
    /// Helper for `encodeByteLevel`. No byte fallback (byte-level vocab is
    /// total over the 256-cp alphabet).
    fn bpeMergeMapped(
        self: *const Tokenizer,
        allocator: Allocator,
        mapped: []const u8,
        out: *std.ArrayList(u32),
    ) TokenizerError!void {
        if (mapped.len == 0) return;

        var symbols: std.ArrayList(Symbol) = .empty;
        defer symbols.deinit(allocator);
        var i: usize = 0;
        while (i < mapped.len) {
            const len = utf8Len(mapped[i]);
            const take = @min(len, mapped.len - i);
            try symbols.append(allocator, .{
                .text = mapped[i .. i + take],
                .prev = @intCast(@as(isize, @intCast(symbols.items.len)) - 1),
                .next = -1,
                .alive = true,
            });
            i += take;
        }
        for (symbols.items, 0..) |*sym, idx| {
            sym.next = if (idx + 1 < symbols.items.len) @intCast(idx + 1) else -1;
        }

        var queue: BigramQueue = .empty;
        defer queue.deinit(allocator);
        for (1..symbols.items.len) |idx| {
            try self.tryAddBigram(allocator, &queue, symbols.items, @intCast(idx - 1), @intCast(idx));
        }
        while (queue.pop()) |bg| {
            const left_idx: usize = @intCast(bg.left);
            const right_idx: usize = @intCast(bg.right);
            const left = &symbols.items[left_idx];
            const right = &symbols.items[right_idx];
            if (!left.alive or !right.alive) continue;
            if (left.text.len + right.text.len != bg.merged_len) continue;
            const start_off = @intFromPtr(left.text.ptr) - @intFromPtr(mapped.ptr);
            const end_off = @intFromPtr(right.text.ptr) - @intFromPtr(mapped.ptr) + right.text.len;
            left.text = mapped[start_off..end_off];
            right.alive = false;
            left.next = right.next;
            if (right.next >= 0) {
                symbols.items[@intCast(right.next)].prev = bg.left;
            }
            try self.tryAddBigram(allocator, &queue, symbols.items, left.prev, bg.left);
            try self.tryAddBigram(allocator, &queue, symbols.items, bg.left, left.next);
        }

        var cur: i32 = if (symbols.items.len == 0) -1 else 0;
        while (cur >= 0) {
            const sym = symbols.items[@intCast(cur)];
            if (sym.alive) {
                if (self.by_piece.get(sym.text)) |hit| {
                    try out.append(allocator, hit.id);
                }
                // No byte fallback. A miss here means the vocab does not
                // contain a single mapped byte the alphabet produced, which
                // would indicate a malformed byte-level tokenizer. Drop
                // silently rather than emit garbage.
            }
            cur = sym.next;
        }
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

    /// Build a Tokenizer from a HuggingFace `tokenizer.json` payload (the
    /// `tokenizers` crate JSON serialization that MLX checkpoints ship).
    /// Targets Llama-family models — BPE with `byte_fallback`, SentencePiece
    /// `▁` (U+2581) word-start marker, and a `Prepend ▁ + Replace ' '→'▁'`
    /// normalizer pipeline. The existing greedy-BPE encoder above runs
    /// unchanged once `pieces` and `scores` are populated correctly.
    pub fn initFromHfJson(allocator: Allocator, json_bytes: []const u8) TokenizerError!Tokenizer {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        const aalloc = arena.allocator();
        defer arena.deinit();

        const parsed = std.json.parseFromSlice(std.json.Value, aalloc, json_bytes, .{}) catch
            return error.InvalidVocabPayload;
        const root = parsed.value;
        if (root != .object) return error.InvalidVocabPayload;

        const model_v = root.object.get("model") orelse return error.MissingVocab;
        if (model_v != .object) return error.InvalidVocabPayload;
        const vocab_v = model_v.object.get("vocab") orelse return error.MissingVocab;
        if (vocab_v != .object) return error.InvalidVocabPayload;
        const merges_v = model_v.object.get("merges");

        // First pass: highest token id, to size pieces/scores.
        var max_id: usize = 0;
        var vit = vocab_v.object.iterator();
        while (vit.next()) |kv| {
            if (kv.value_ptr.* != .integer) return error.InvalidVocabPayload;
            const id = kv.value_ptr.*.integer;
            if (id < 0) return error.InvalidVocabPayload;
            const id_us: usize = @intCast(id);
            if (id_us > max_id) max_id = id_us;
        }
        // Specials go through `added_tokens`; include them in the size.
        if (root.object.get("added_tokens")) |at_v| {
            if (at_v == .array) {
                for (at_v.array.items) |entry| {
                    if (entry != .object) continue;
                    if (entry.object.get("id")) |id_v| {
                        if (id_v == .integer and id_v.integer >= 0) {
                            const id_us: usize = @intCast(id_v.integer);
                            if (id_us > max_id) max_id = id_us;
                        }
                    }
                }
            }
        }
        const count = max_id + 1;

        const pieces = try allocator.alloc([]const u8, count);
        errdefer allocator.free(pieces);
        for (pieces) |*p| p.* = "";
        const scores = try allocator.alloc(f32, count);
        errdefer allocator.free(scores);
        // Default score so non-merge tokens never beat an actual merge in the
        // bigram queue. `-1e9` is well below any merge index even for huge
        // vocabularies.
        @memset(scores, -1e9);

        // Pieces: vocab. Strings live in arena; copy out so they outlive the
        // arena's deinit at function exit.
        vit = vocab_v.object.iterator();
        while (vit.next()) |kv| {
            const id_us: usize = @intCast(kv.value_ptr.*.integer);
            pieces[id_us] = try allocator.dupe(u8, kv.key_ptr.*);
        }

        // Detect tokenizer flavor from pre_tokenizer.type. ByteLevel (raw
        // or nested in a Sequence) → byte-level pipeline; everything else
        // (default, Metaspace, etc.) keeps the SentencePiece path.
        // Also sniff for the cl100k_base contraction signature in any
        // nested Split's regex pattern → use splitLlama3.
        var flavor: Flavor = .sentencepiece;
        var byte_split: ByteSplit = .gpt2;
        if (root.object.get("pre_tokenizer")) |pt_v| {
            flavor = detectFlavor(pt_v);
            byte_split = detectByteSplit(pt_v);
        }

        // added_tokens override (e.g., `<unk>`, `<s>`, `</s>`).
        var bos: u32 = 1;
        var eos: u32 = 2;
        if (root.object.get("added_tokens")) |at_v| {
            if (at_v == .array) {
                for (at_v.array.items) |entry| {
                    if (entry != .object) continue;
                    const id_v = entry.object.get("id") orelse continue;
                    const content_v = entry.object.get("content") orelse continue;
                    if (id_v != .integer or content_v != .string) continue;
                    const id_us: usize = @intCast(id_v.integer);
                    // Free any vocab-derived dupe at this id before overwriting:
                    // added_tokens like `<s>` / `</s>` / `<unk>` shadow the
                    // earlier vocab entry, and skipping the free leaks N
                    // tokens worth of memory per call (caught by DebugAllocator
                    // in the integration test).
                    if (pieces[id_us].len > 0) allocator.free(pieces[id_us]);
                    pieces[id_us] = try allocator.dupe(u8, content_v.string);
                    if (std.mem.eql(u8, content_v.string, "<s>")) bos = @intCast(id_us);
                    if (std.mem.eql(u8, content_v.string, "</s>")) eos = @intCast(id_us);
                }
            }
        }

        // Scores: assign higher score to earlier merges so the greedy BPE
        // encoder picks them first. Format is either a list of "A B" strings
        // or a list of [A, B] arrays — handle both.
        if (merges_v) |mv| if (mv == .array) {
            for (mv.array.items, 0..) |m_entry, i| {
                var a_str: []const u8 = "";
                var b_str: []const u8 = "";
                switch (m_entry) {
                    .string => |s| {
                        const sp = std.mem.indexOfScalar(u8, s, ' ') orelse continue;
                        a_str = s[0..sp];
                        b_str = s[sp + 1 ..];
                    },
                    .array => |arr| {
                        if (arr.items.len != 2) continue;
                        if (arr.items[0] != .string or arr.items[1] != .string) continue;
                        a_str = arr.items[0].string;
                        b_str = arr.items[1].string;
                    },
                    else => continue,
                }
                // Concatenate A+B and look up the resulting token's id.
                var join_buf: [512]u8 = undefined;
                if (a_str.len + b_str.len > join_buf.len) continue;
                @memcpy(join_buf[0..a_str.len], a_str);
                @memcpy(join_buf[a_str.len..][0..b_str.len], b_str);
                const joined = join_buf[0 .. a_str.len + b_str.len];
                const id_v = vocab_v.object.get(joined) orelse continue;
                if (id_v != .integer) continue;
                const id_us: usize = @intCast(id_v.integer);
                if (id_us < scores.len) {
                    scores[id_us] = -@as(f32, @floatFromInt(i));
                }
            }
        };

        // Build piece map for O(1) lookup.
        var by_piece: PieceMap = .empty;
        errdefer by_piece.deinit(allocator);
        try by_piece.ensureTotalCapacity(allocator, @intCast(count));
        for (pieces, scores, 0..) |piece, score, i| {
            if (piece.len == 0) continue;
            try by_piece.put(allocator, piece, .{ .id = @intCast(i), .score = score });
        }

        return .{
            .pieces = pieces,
            .scores = scores,
            .by_piece = by_piece,
            .bos = bos,
            .eos = eos,
            .flavor = flavor,
            .byte_split = byte_split,
        };
    }

    /// Like `deinit`, but additionally frees the per-piece arena-allocated
    /// strings (HF init duplicates them out of the arena, so plain `deinit`
    /// would leak). Use this when the Tokenizer was built with
    /// `initFromHfJson`.
    pub fn deinitOwnedPieces(self: *Tokenizer, allocator: Allocator) void {
        // Pieces with len == 0 are the static "" literal placed at vocab
        // gaps in `initFromHfJson`; freeing them would be UB. Only free the
        // non-empty entries that actually came from `allocator.dupe`.
        for (self.pieces) |p| {
            if (p.len > 0) allocator.free(p);
        }
        self.by_piece.deinit(allocator);
        allocator.free(self.scores);
        allocator.free(self.pieces);
        self.* = undefined;
    }

    /// Write the decoded bytes for `token_id` into `writer`. Dispatches on
    /// `self.flavor`:
    ///  * SentencePiece: ▁ → space, <0xHH> → raw byte.
    ///  * Byte-level: walk the piece's codepoints, reverse the GPT-2
    ///    byte→Unicode alphabet to recover the original bytes.
    pub fn decodeTo(self: *const Tokenizer, writer: *std.Io.Writer, token_id: u32) !void {
        if (token_id >= self.pieces.len) return;
        const piece = self.pieces[token_id];
        switch (self.flavor) {
            .sentencepiece => try decodeSentencePiece(writer, piece),
            .byte_level => try decodeByteLevel(writer, piece),
        }
    }
};

fn decodeSentencePiece(writer: *std.Io.Writer, piece: []const u8) !void {
    if (piece.len == 6 and piece[0] == '<' and piece[1] == '0' and piece[2] == 'x' and piece[5] == '>') {
        const hi = hexNibble(piece[3]) orelse return writer.writeAll(piece);
        const lo = hexNibble(piece[4]) orelse return writer.writeAll(piece);
        try writer.writeByte(@intCast((hi << 4) | lo));
        return;
    }
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

fn decodeByteLevel(writer: *std.Io.Writer, piece: []const u8) !void {
    var i: usize = 0;
    while (i < piece.len) {
        const d = byte_level_bpe.utf8Decode(piece[i..]);
        if (byte_level_bpe.inverse.get(d.cp)) |b| {
            try writer.writeByte(b);
        } else {
            // Codepoint outside the alphabet; emit raw UTF-8 bytes
            // verbatim. Should not happen for a well-formed byte-level
            // tokenizer, but degrades gracefully if a piece slips through.
            try writer.writeAll(piece[i .. i + d.len]);
        }
        i += d.len;
    }
}

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

/// Walk a `pre_tokenizer` JSON node looking for a Split with the cl100k
/// contraction signature (`'s|'t|'re`). Returns `.llama3` when found, else
/// `.gpt2`. Reliable across the cl100k variants shipped by Qwen2 / Llama-3
/// / Mistral-NeMo since they all carry the same regex byte-for-byte.
fn detectByteSplit(node: std.json.Value) ByteSplit {
    switch (node) {
        .object => |obj| {
            if (obj.get("type")) |t_v| {
                if (t_v == .string and std.mem.eql(u8, t_v.string, "Split")) {
                    if (obj.get("pattern")) |pat_v| {
                        if (pat_v == .object) {
                            if (pat_v.object.get("Regex")) |r_v| {
                                if (r_v == .string and std.mem.indexOf(u8, r_v.string, "'s|'t|'re") != null) {
                                    return .llama3;
                                }
                            }
                        }
                    }
                }
            }
            if (obj.get("pretokenizers")) |inner| {
                if (inner == .array) {
                    for (inner.array.items) |child| {
                        if (detectByteSplit(child) == .llama3) return .llama3;
                    }
                }
            }
        },
        else => {},
    }
    return .gpt2;
}

/// Walk a `pre_tokenizer` JSON node looking for a `ByteLevel` type marker
/// — either as a leaf or nested in a Sequence/list. Returns `.byte_level`
/// when found, `.sentencepiece` otherwise.
fn detectFlavor(node: std.json.Value) Flavor {
    switch (node) {
        .object => |obj| {
            if (obj.get("type")) |t_v| {
                if (t_v == .string and std.mem.eql(u8, t_v.string, "ByteLevel")) {
                    return .byte_level;
                }
            }
            // Sequence with nested pretokenizers: scan the inner array.
            if (obj.get("pretokenizers")) |inner| {
                if (inner == .array) {
                    for (inner.array.items) |child| {
                        if (detectFlavor(child) == .byte_level) return .byte_level;
                    }
                }
            }
        },
        else => {},
    }
    return .sentencepiece;
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

test "byte-level: encode + decode round-trips ASCII text on synthetic vocab" {
    const allocator = std.testing.allocator;
    // Build a minimal byte-level vocab covering every byte 0..255 as a
    // single-codepoint piece. No merges: every output token is one byte.
    const piece_count: usize = 256;
    const pieces = try allocator.alloc([]const u8, piece_count);
    defer {
        for (pieces) |p| allocator.free(p);
        allocator.free(pieces);
    }
    for (pieces, 0..) |*p, b| {
        var buf: [4]u8 = undefined;
        const n = std.unicode.utf8Encode(byte_level_bpe.byte_to_unicode[b], &buf) catch unreachable;
        p.* = try allocator.dupe(u8, buf[0..n]);
    }
    const scores = try allocator.alloc(f32, piece_count);
    defer allocator.free(scores);
    @memset(scores, 0);

    var by_piece: PieceMap = .empty;
    defer by_piece.deinit(allocator);
    try by_piece.ensureTotalCapacity(allocator, @intCast(piece_count));
    for (pieces, 0..) |p, i| {
        try by_piece.put(allocator, p, .{ .id = @intCast(i), .score = 0 });
    }

    const tok: Tokenizer = .{
        .pieces = pieces,
        .scores = scores,
        .by_piece = by_piece,
        .bos = 0,
        .eos = 0,
        .flavor = .byte_level,
    };

    const original = "Hello, world! 123\n\tend.";
    const ids = try tok.encode(allocator, original, false);
    defer allocator.free(ids);
    // No merges → one token per byte.
    try std.testing.expectEqual(original.len, ids.len);

    var buf: [128]u8 = undefined;
    var fbw: std.Io.Writer = .fixed(&buf);
    for (ids) |id| try tok.decodeTo(&fbw, id);
    try std.testing.expectEqualSlices(u8, original, fbw.buffered());
}

test "detectFlavor: ByteLevel pre_tokenizer → byte_level" {
    const json_byte = "{\"type\":\"ByteLevel\",\"add_prefix_space\":false}";
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const v = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), json_byte, .{});
    try std.testing.expectEqual(Flavor.byte_level, detectFlavor(v));
}

test "detectFlavor: Sequence with ByteLevel inner → byte_level" {
    const json_byte =
        "{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Split\"},{\"type\":\"ByteLevel\"}]}";
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const v = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), json_byte, .{});
    try std.testing.expectEqual(Flavor.byte_level, detectFlavor(v));
}

test "detectByteSplit: cl100k regex in nested Split → llama3" {
    const json_byte =
        "{\"type\":\"Sequence\",\"pretokenizers\":[" ++
        "{\"type\":\"Split\",\"pattern\":{\"Regex\":\"(?i:'s|'t|'re)|...\"}}," ++
        "{\"type\":\"ByteLevel\"}]}";
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const v = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), json_byte, .{});
    try std.testing.expectEqual(ByteSplit.llama3, detectByteSplit(v));
}

test "detectByteSplit: GPT-2 ByteLevel without cl100k regex → gpt2" {
    const json_byte = "{\"type\":\"ByteLevel\",\"add_prefix_space\":false}";
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const v = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), json_byte, .{});
    try std.testing.expectEqual(ByteSplit.gpt2, detectByteSplit(v));
}

test "detectFlavor: Metaspace → sentencepiece" {
    const json_byte = "{\"type\":\"Metaspace\",\"replacement\":\"_\"}";
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const v = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), json_byte, .{});
    try std.testing.expectEqual(Flavor.sentencepiece, detectFlavor(v));
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
