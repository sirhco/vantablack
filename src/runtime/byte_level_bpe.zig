//! Tiktoken-style byte-level BPE encoder.
//!
//! Two pieces beyond the existing SentencePiece BPE engine:
//!   1. The GPT-2 byte→Unicode alphabet — every input byte maps to a
//!      printable Unicode code point so BPE merges run on a string with no
//!      embedded NUL / control bytes.
//!   2. A Unicode-class-aware pre-tokenizer that splits text into runs of
//!      letters / numbers / non-letter-non-number-non-space / spaces
//!      (the GPT-2 split pattern). Runs become independent merge contexts.
//!
//! Llama-3's contraction-aware split (`(?i:'s|'t|...)`) is NOT yet
//! implemented — when fixtures are available it can be added as
//! `splitLlama3` next to `splitGpt2`.
//!
//! The merge stage reuses the existing greedy BPE engine via
//! `Tokenizer.mergeText`, which already drives SentencePiece encoding.

const std = @import("std");
const Allocator = std.mem.Allocator;

const unicode_table = @import("unicode_table.zig");

pub const Span = struct { start: usize, end: usize };

pub const Splitter = *const fn (text: []const u8, out: *std.ArrayList(Span), allocator: Allocator) anyerror!void;

/// Build the GPT-2 byte→Unicode alphabet at comptime.
/// Reference: https://github.com/openai/gpt-2/blob/master/src/encoder.py::bytes_to_unicode
pub const byte_to_unicode: [256]u21 = blk: {
    @setEvalBranchQuota(20000);
    var map: [256]u21 = undefined;
    var n: u21 = 0;
    var b: u21 = 0;
    while (b < 256) : (b += 1) {
        const printable = (b >= 33 and b <= 126) or
            (b >= 161 and b <= 172) or
            (b >= 174 and b <= 255);
        if (printable) {
            map[b] = b;
        } else {
            map[b] = 256 + n;
            n += 1;
        }
    }
    break :blk map;
};

/// Reverse map for decode. unicode_to_byte[map[b]] == b.
/// Built lazily into a sparse hash because the codomain is 0..324.
pub const UnicodeToByte = struct {
    // Direct lookup — codomain max is 32 + 256 + 35 = ~324 in practice.
    // Sized 512 for headroom + faster modulo-free indexing.
    table: [512]i16, // -1 = not in map; otherwise byte value 0..255

    pub fn init() UnicodeToByte {
        var t: [512]i16 = @splat(-1);
        for (byte_to_unicode, 0..) |cp, b| {
            if (cp < 512) t[cp] = @intCast(b);
        }
        return .{ .table = t };
    }

    pub fn get(self: UnicodeToByte, cp: u21) ?u8 {
        if (cp >= 512) return null;
        const v = self.table[cp];
        if (v < 0) return null;
        return @intCast(v);
    }
};

pub const inverse = UnicodeToByte.init();

/// Encode codepoint as UTF-8, append to `out`. cp must be < 0x110000.
pub fn utf8Append(cp: u21, out: *std.ArrayList(u8), allocator: Allocator) !void {
    var buf: [4]u8 = undefined;
    const n = std.unicode.utf8Encode(cp, &buf) catch return error.InvalidUtf8;
    try out.appendSlice(allocator, buf[0..n]);
}

const Decoded = struct { cp: u21, len: u3 };

/// Decode the leading UTF-8 codepoint in `text`. Treats malformed leads
/// as a single byte for resilience.
pub fn utf8Decode(text: []const u8) Decoded {
    if (text.len == 0) return .{ .cp = 0, .len = 1 };
    const first = text[0];
    if (first < 0x80) return .{ .cp = first, .len = 1 };
    const len: u3 = if ((first & 0xE0) == 0xC0) 2 else if ((first & 0xF0) == 0xE0) 3 else if ((first & 0xF8) == 0xF0) 4 else 1;
    if (len == 1 or text.len < len) return .{ .cp = first, .len = 1 };
    const cp = std.unicode.utf8Decode(text[0..len]) catch return .{ .cp = first, .len = 1 };
    return .{ .cp = cp, .len = len };
}

/// Lower-case ASCII contractions Llama-3 / cl100k_base recognise as
/// stand-alone pre-tokens. The reference regex is case-insensitive; the
/// match function below applies `b | 0x20` to letter bytes so 'S / 't / 'RE
/// behave identically.
const llama3_contractions = [_][]const u8{ "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };

/// Try to match a Llama-3 contraction at `text[start..]` case-insensitively.
/// Returns the contraction byte length if matched, otherwise null.
fn matchContraction(text: []const u8, start: usize) ?usize {
    if (start >= text.len or text[start] != '\'') return null;
    for (llama3_contractions) |c| {
        if (start + c.len > text.len) continue;
        var ok = true;
        for (c, 0..) |ec, i| {
            const tb = text[start + i];
            const tc = if (tb >= 'A' and tb <= 'Z') tb | 0x20 else tb;
            if (tc != ec) {
                ok = false;
                break;
            }
        }
        if (ok) return c.len;
    }
    return null;
}

/// Llama-3 / cl100k-style pre-tokenizer split. Implements the OpenAI
/// cl100k_base regex shipped with Qwen2 / Llama-3 / Mistral-NeMo
/// tokenizer.json files, alternative-by-alternative:
///
///   1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
///   2. [^\r\n\p{L}\p{N}]?\p{L}+      — single optional non-letter-non-num
///                                       leading char (incl. space) + letter run
///   3. \p{N}{1,3}                    — 1-3 digits, no leading binding
///   4.  ?[^\s\p{L}\p{N}]+[\r\n]*     — optional ASCII space + non-LN-non-space
///                                       run + trailing CR/LF*
///   5. \s*[\r\n]+                    — whitespace ending in newline(s)
///   6. \s+(?!\S)                     — whitespace run, when followed by
///                                       another non-whitespace, peels all
///                                       but the last space (which binds to
///                                       the next rule's leading char)
///   7. \s+                            — pure trailing whitespace fallback
///
/// Verified span-for-span vs the Python `regex` reference compiled from
/// the actual Qwen2 / Llama-3 cl100k pattern on a fixture set including
/// "Hello, world!", "It's mine", "I'M he'D", "Hello world", "12345",
/// "abc 123 xyz", "  hi", "end.\\n". Vocab IDs need a separate parity run
/// vs HF `tokenizers` once a Llama-3 vocab is loaded.
pub fn splitLlama3(text: []const u8, out: *std.ArrayList(Span), allocator: Allocator) !void {
    var i: usize = 0;
    while (i < text.len) {
        const start = i;

        // 1. Contractions.
        if (matchContraction(text, i)) |n| {
            i += n;
            try out.append(allocator, .{ .start = start, .end = i });
            continue;
        }

        const first = utf8Decode(text[i..]);
        const first_cat = unicode_table.category(first.cp);

        // 2. Letters with optional non-LN-non-newline leading char.
        if (first_cat == .letter) {
            i += first.len;
            while (i < text.len) {
                const n = utf8Decode(text[i..]);
                if (unicode_table.category(n.cp) != .letter) break;
                i += n.len;
            }
            try out.append(allocator, .{ .start = start, .end = i });
            continue;
        }
        if (first.cp != '\r' and first.cp != '\n') {
            // Optional leading non-LN char must be followed by at least one letter.
            const head = i + first.len;
            if (head < text.len) {
                const after = utf8Decode(text[head..]);
                if (unicode_table.category(after.cp) == .letter) {
                    i = head + after.len;
                    while (i < text.len) {
                        const n = utf8Decode(text[i..]);
                        if (unicode_table.category(n.cp) != .letter) break;
                        i += n.len;
                    }
                    try out.append(allocator, .{ .start = start, .end = i });
                    continue;
                }
            }
        }

        // 3. Digits 1-3, no leading binding.
        if (first_cat == .number) {
            i += first.len;
            var digits: usize = 1;
            while (i < text.len and digits < 3) {
                const n = utf8Decode(text[i..]);
                if (unicode_table.category(n.cp) != .number) break;
                i += n.len;
                digits += 1;
            }
            try out.append(allocator, .{ .start = start, .end = i });
            continue;
        }

        // 4. " ?[^\s\p{L}\p{N}]+[\r\n]*"
        const punct_head: usize = if (first.cp == ' ') i + 1 else i;
        if (punct_head < text.len) {
            const body = utf8Decode(text[punct_head..]);
            const body_cat = unicode_table.category(body.cp);
            if (body_cat != .letter and body_cat != .number and body_cat != .space) {
                i = punct_head + body.len;
                while (i < text.len) {
                    const n = utf8Decode(text[i..]);
                    const c = unicode_table.category(n.cp);
                    if (c == .letter or c == .number or c == .space) break;
                    i += n.len;
                }
                while (i < text.len and (text[i] == '\r' or text[i] == '\n')) i += 1;
                try out.append(allocator, .{ .start = start, .end = i });
                continue;
            }
        }

        // 5/6/7. Whitespace.
        var ws_end = i;
        while (ws_end < text.len) {
            const n = utf8Decode(text[ws_end..]);
            if (unicode_table.category(n.cp) != .space) break;
            ws_end += n.len;
        }
        if (ws_end == i) {
            // Unrecognized byte — single-byte span for forward progress.
            i = i + 1;
            try out.append(allocator, .{ .start = start, .end = i });
            continue;
        }
        var has_newline = false;
        var k: usize = i;
        while (k < ws_end) : (k += 1) {
            if (text[k] == '\r' or text[k] == '\n') {
                has_newline = true;
                break;
            }
        }
        if (has_newline or ws_end == text.len) {
            // Rule 5 (newline-terminated) or rule 7 (trailing whitespace at EOF).
            try out.append(allocator, .{ .start = start, .end = ws_end });
            i = ws_end;
            continue;
        }
        // Rule 6: whitespace followed by non-whitespace. Peel all but the
        // last space; the trailing space binds to the next rule's optional
        // leading char on the following loop iteration.
        const peeled = ws_end - 1;
        if (peeled > i) {
            try out.append(allocator, .{ .start = start, .end = peeled });
            i = peeled;
        } else {
            // ws_end - i == 1 — single space before non-space. cl100k
            // treats this as standalone " " span (the next pre-token
            // doesn't bind across the boundary because there's nothing to
            // peel before it).
            try out.append(allocator, .{ .start = start, .end = ws_end });
            i = ws_end;
        }
    }
}

/// GPT-2 pre-tokenizer split: groups consecutive codepoints by Unicode
/// category (letter / number / space / other). Emits one Span per run.
pub fn splitGpt2(text: []const u8, out: *std.ArrayList(Span), allocator: Allocator) !void {
    var i: usize = 0;
    while (i < text.len) {
        const start = i;
        const first = utf8Decode(text[i..]);
        const cat = unicode_table.category(first.cp);
        i += first.len;
        while (i < text.len) {
            const next = utf8Decode(text[i..]);
            if (unicode_table.category(next.cp) != cat) break;
            i += next.len;
        }
        try out.append(allocator, .{ .start = start, .end = i });
    }
}

test "byte_to_unicode: spot checks" {
    try std.testing.expectEqual(@as(u21, 'a'), byte_to_unicode[97]);
    try std.testing.expectEqual(@as(u21, 'A'), byte_to_unicode[65]);
    try std.testing.expectEqual(@as(u21, '0'), byte_to_unicode[48]);
    // Space (0x20) is non-printable in GPT-2's classification — should map
    // to 0x100 + (32 - 0) since bytes 0..32 are the first non-printable run.
    try std.testing.expectEqual(@as(u21, 0x100 + 32), byte_to_unicode[32]);
    // Tab (9) → 0x100 + 9.
    try std.testing.expectEqual(@as(u21, 0x100 + 9), byte_to_unicode[9]);
    // 0xFF identity (in the third printable run).
    try std.testing.expectEqual(@as(u21, 0xFF), byte_to_unicode[0xFF]);
}

test "byte_to_unicode is injective (every codepoint unique)" {
    var seen: std.AutoHashMap(u21, void) = .init(std.testing.allocator);
    defer seen.deinit();
    for (byte_to_unicode) |cp| {
        try std.testing.expect(!seen.contains(cp));
        try seen.put(cp, {});
    }
}

test "inverse: round-trips every byte" {
    for (0..256) |b| {
        const cp = byte_to_unicode[b];
        const back = inverse.get(cp).?;
        try std.testing.expectEqual(@as(u8, @intCast(b)), back);
    }
}

test "splitGpt2: alpha + space + alpha + punct" {
    var splits: std.ArrayList(Span) = .empty;
    defer splits.deinit(std.testing.allocator);
    try splitGpt2("Hello world!", &splits, std.testing.allocator);
    // letters + space + letters + other(!)
    try std.testing.expectEqual(@as(usize, 4), splits.items.len);
    try std.testing.expectEqualSlices(u8, "Hello", "Hello world!"[splits.items[0].start..splits.items[0].end]);
    try std.testing.expectEqualSlices(u8, " ", "Hello world!"[splits.items[1].start..splits.items[1].end]);
    try std.testing.expectEqualSlices(u8, "world", "Hello world!"[splits.items[2].start..splits.items[2].end]);
    try std.testing.expectEqualSlices(u8, "!", "Hello world!"[splits.items[3].start..splits.items[3].end]);
}

test "splitGpt2: mixed letters/numbers/whitespace" {
    var splits: std.ArrayList(Span) = .empty;
    defer splits.deinit(std.testing.allocator);
    try splitGpt2("abc 123\t\nxyz", &splits, std.testing.allocator);
    // letters, space, number, space (\t\n), letters
    try std.testing.expectEqual(@as(usize, 5), splits.items.len);
    try std.testing.expectEqualSlices(u8, "abc", "abc 123\t\nxyz"[splits.items[0].start..splits.items[0].end]);
    try std.testing.expectEqualSlices(u8, " ", "abc 123\t\nxyz"[splits.items[1].start..splits.items[1].end]);
    try std.testing.expectEqualSlices(u8, "123", "abc 123\t\nxyz"[splits.items[2].start..splits.items[2].end]);
    try std.testing.expectEqualSlices(u8, "\t\n", "abc 123\t\nxyz"[splits.items[3].start..splits.items[3].end]);
    try std.testing.expectEqualSlices(u8, "xyz", "abc 123\t\nxyz"[splits.items[4].start..splits.items[4].end]);
}

test "splitGpt2: empty input → no splits" {
    var splits: std.ArrayList(Span) = .empty;
    defer splits.deinit(std.testing.allocator);
    try splitGpt2("", &splits, std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), splits.items.len);
}

test "splitLlama3: contractions split as standalone pre-tokens" {
    var splits: std.ArrayList(Span) = .empty;
    defer splits.deinit(std.testing.allocator);
    const text = "It's mine";
    try splitLlama3(text, &splits, std.testing.allocator);
    // Expected: ["It", "'s", " mine"]
    try std.testing.expectEqual(@as(usize, 3), splits.items.len);
    try std.testing.expectEqualSlices(u8, "It", text[splits.items[0].start..splits.items[0].end]);
    try std.testing.expectEqualSlices(u8, "'s", text[splits.items[1].start..splits.items[1].end]);
    try std.testing.expectEqualSlices(u8, " mine", text[splits.items[2].start..splits.items[2].end]);
}

test "splitLlama3: case-insensitive contractions" {
    var splits: std.ArrayList(Span) = .empty;
    defer splits.deinit(std.testing.allocator);
    const text = "I'M he'D";
    try splitLlama3(text, &splits, std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 4), splits.items.len);
    try std.testing.expectEqualSlices(u8, "I",   text[splits.items[0].start..splits.items[0].end]);
    try std.testing.expectEqualSlices(u8, "'M",  text[splits.items[1].start..splits.items[1].end]);
    try std.testing.expectEqualSlices(u8, " he", text[splits.items[2].start..splits.items[2].end]);
    try std.testing.expectEqualSlices(u8, "'D",  text[splits.items[3].start..splits.items[3].end]);
}

test "splitLlama3: leading-space binding keeps ' word' together" {
    var splits: std.ArrayList(Span) = .empty;
    defer splits.deinit(std.testing.allocator);
    const text = "Hello world";
    try splitLlama3(text, &splits, std.testing.allocator);
    // splitGpt2 would emit ["Hello", " ", "world"]; splitLlama3 emits
    // ["Hello", " world"].
    try std.testing.expectEqual(@as(usize, 2), splits.items.len);
    try std.testing.expectEqualSlices(u8, "Hello", text[splits.items[0].start..splits.items[0].end]);
    try std.testing.expectEqualSlices(u8, " world", text[splits.items[1].start..splits.items[1].end]);
}

test "splitLlama3: digits chunk 1-3 at a time" {
    var splits: std.ArrayList(Span) = .empty;
    defer splits.deinit(std.testing.allocator);
    const text = "12345";
    try splitLlama3(text, &splits, std.testing.allocator);
    // 5 digits -> "123" + "45" (2 spans).
    try std.testing.expectEqual(@as(usize, 2), splits.items.len);
    try std.testing.expectEqualSlices(u8, "123", text[splits.items[0].start..splits.items[0].end]);
    try std.testing.expectEqualSlices(u8, "45",  text[splits.items[1].start..splits.items[1].end]);
}

fn expectSpans(text: []const u8, expected: []const []const u8) !void {
    var splits: std.ArrayList(Span) = .empty;
    defer splits.deinit(std.testing.allocator);
    try splitLlama3(text, &splits, std.testing.allocator);
    try std.testing.expectEqual(expected.len, splits.items.len);
    for (splits.items, expected) |sp, exp| {
        try std.testing.expectEqualSlices(u8, exp, text[sp.start..sp.end]);
    }
}

test "splitLlama3: matches cl100k Python regex on fixture set" {
    // Reference outputs from `python -c "import regex; ..."` against the
    // actual Qwen2 / Llama-3 cl100k_base pattern.
    try expectSpans("Hello, world!", &.{ "Hello", ",", " world", "!" });
    try expectSpans("Hello world",   &.{ "Hello", " world" });
    try expectSpans("It's mine",     &.{ "It", "'s", " mine" });
    try expectSpans("I'M he'D",      &.{ "I", "'M", " he", "'D" });
    try expectSpans("12345",         &.{ "123", "45" });
    try expectSpans("abc 123 xyz",   &.{ "abc", " ", "123", " xyz" });
    try expectSpans("end.\n",        &.{ "end", ".\n" });
}

test "splitLlama3: whitespace peel binds trailing space to next pre-token" {
    // Reference: "  hi" -> [" ", " hi"]. The first space is standalone
    // (rule 6 peels run minus one); the second binds to "hi" via rule 2.
    try expectSpans("  hi", &.{ " ", " hi" });
    // Triple space: peel two, last binds.
    try expectSpans("   hi", &.{ "  ", " hi" });
}

test "splitLlama3: covers every input byte (no gaps)" {
    var splits: std.ArrayList(Span) = .empty;
    defer splits.deinit(std.testing.allocator);
    const text = "Hello, world! 1234 'tis fine.\n";
    try splitLlama3(text, &splits, std.testing.allocator);
    var covered: usize = 0;
    for (splits.items) |sp| {
        try std.testing.expectEqual(covered, sp.start);
        covered = sp.end;
    }
    try std.testing.expectEqual(text.len, covered);
}

test "byte-encode + UTF-8 round trip" {
    // For every byte sequence, byte_to_unicode → utf8Append → utf8Decode
    // → inverse.get must reproduce the original byte.
    const allocator = std.testing.allocator;
    var encoded: std.ArrayList(u8) = .empty;
    defer encoded.deinit(allocator);
    const original = "Hello, world!\nLine 2\twith\t\ttabs.";
    for (original) |b| try utf8Append(byte_to_unicode[b], &encoded, allocator);
    // Decode back.
    var decoded: std.ArrayList(u8) = .empty;
    defer decoded.deinit(allocator);
    var i: usize = 0;
    while (i < encoded.items.len) {
        const d = utf8Decode(encoded.items[i..]);
        const b = inverse.get(d.cp) orelse return error.RoundTripFailed;
        try decoded.append(allocator, b);
        i += d.len;
    }
    try std.testing.expectEqualSlices(u8, original, decoded.items);
}
