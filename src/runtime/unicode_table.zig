//! Compact Unicode general-category lookup for the GPT-2 / Llama-3
//! pre-tokenizer regex (`\p{L}`, `\p{N}`, `\s`, otherwise).
//!
//! Coverage is pragmatic, not Unicode-complete: ASCII plus the common
//! scripts shipped on real-world prompts (Latin-1, Latin Extended A/B,
//! Greek, Cyrillic, CJK Unified Ideographs basic + Extension A,
//! Hiragana, Katakana, Hangul Syllables). Anything outside these ranges
//! falls into `.other`. Round-trip safety is unaffected (decode walks the
//! reverse byte→Unicode map directly), but tokenization of multilingual
//! prompts whose scripts aren't listed may split sub-optimally.
//!
//! Re-vendoring against newer Unicode versions: replace the range arrays
//! below from `UnicodeData.txt` (categories L* / N*).

const std = @import("std");

pub const Category = enum(u3) { letter, number, space, other };

const Range = struct { lo: u21, hi: u21 };

// Letter ranges (Lu, Ll, Lt, Lm, Lo). Ordered ascending so binary search
// is deterministic. Hand-curated from Unicode 15.x core blocks.
const letter_ranges = [_]Range{
    .{ .lo = 'A', .hi = 'Z' },
    .{ .lo = 'a', .hi = 'z' },
    .{ .lo = 0x00AA, .hi = 0x00AA }, // FEMININE ORDINAL INDICATOR
    .{ .lo = 0x00B5, .hi = 0x00B5 }, // MICRO SIGN
    .{ .lo = 0x00BA, .hi = 0x00BA }, // MASCULINE ORDINAL INDICATOR
    .{ .lo = 0x00C0, .hi = 0x00D6 }, // Latin-1 letters (skip multiplication sign at D7)
    .{ .lo = 0x00D8, .hi = 0x00F6 },
    .{ .lo = 0x00F8, .hi = 0x024F }, // Latin Extended-A + B
    .{ .lo = 0x0370, .hi = 0x0373 }, // Greek + Coptic letters (subset)
    .{ .lo = 0x0376, .hi = 0x0377 },
    .{ .lo = 0x037B, .hi = 0x037D },
    .{ .lo = 0x037F, .hi = 0x037F },
    .{ .lo = 0x0386, .hi = 0x0386 },
    .{ .lo = 0x0388, .hi = 0x038A },
    .{ .lo = 0x038C, .hi = 0x038C },
    .{ .lo = 0x038E, .hi = 0x03A1 },
    .{ .lo = 0x03A3, .hi = 0x03F5 },
    .{ .lo = 0x03F7, .hi = 0x0481 },
    .{ .lo = 0x048A, .hi = 0x052F }, // Cyrillic (full block)
    .{ .lo = 0x0531, .hi = 0x0556 }, // Armenian
    .{ .lo = 0x0561, .hi = 0x0587 },
    .{ .lo = 0x05D0, .hi = 0x05EA }, // Hebrew letters
    .{ .lo = 0x0620, .hi = 0x064A }, // Arabic letters (subset)
    .{ .lo = 0x066E, .hi = 0x066F },
    .{ .lo = 0x0671, .hi = 0x06D3 },
    .{ .lo = 0x06D5, .hi = 0x06D5 },
    .{ .lo = 0x0904, .hi = 0x0939 }, // Devanagari letters
    .{ .lo = 0x0E01, .hi = 0x0E30 }, // Thai letters
    .{ .lo = 0x0E32, .hi = 0x0E33 },
    .{ .lo = 0x1100, .hi = 0x11FF }, // Hangul Jamo
    .{ .lo = 0x3041, .hi = 0x3096 }, // Hiragana
    .{ .lo = 0x30A1, .hi = 0x30FA }, // Katakana
    .{ .lo = 0x3400, .hi = 0x4DBF }, // CJK Unified Ideographs Extension A
    .{ .lo = 0x4E00, .hi = 0x9FFF }, // CJK Unified Ideographs
    .{ .lo = 0xAC00, .hi = 0xD7A3 }, // Hangul Syllables
    .{ .lo = 0xF900, .hi = 0xFAFF }, // CJK Compatibility Ideographs
    .{ .lo = 0xFB00, .hi = 0xFB06 }, // Latin small ligatures
};

// Number ranges (Nd / Nl / No: digit-like). Pragmatic subset.
const number_ranges = [_]Range{
    .{ .lo = '0', .hi = '9' },
    .{ .lo = 0x00B2, .hi = 0x00B3 }, // SUPERSCRIPT 2/3
    .{ .lo = 0x00B9, .hi = 0x00B9 }, // SUPERSCRIPT 1
    .{ .lo = 0x00BC, .hi = 0x00BE }, // VULGAR FRACTIONS
    .{ .lo = 0x0660, .hi = 0x0669 }, // Arabic-Indic digits
    .{ .lo = 0x06F0, .hi = 0x06F9 }, // Extended Arabic-Indic digits
    .{ .lo = 0x0966, .hi = 0x096F }, // Devanagari digits
    .{ .lo = 0xFF10, .hi = 0xFF19 }, // Fullwidth digits
};

fn rangeContains(ranges: []const Range, cp: u21) bool {
    var lo: usize = 0;
    var hi: usize = ranges.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const r = ranges[mid];
        if (cp < r.lo) {
            hi = mid;
        } else if (cp > r.hi) {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    return false;
}

pub fn category(cp: u21) Category {
    // Whitespace: ASCII + NBSP + a few common Unicode spaces.
    switch (cp) {
        ' ', '\t', '\n', '\r', 0x0B, 0x0C => return .space,
        0x00A0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => return .space,
        else => {},
    }
    if (cp >= 0x2000 and cp <= 0x200A) return .space;
    if (rangeContains(&number_ranges, cp)) return .number;
    if (rangeContains(&letter_ranges, cp)) return .letter;
    return .other;
}

test "category: ASCII letters and digits" {
    try std.testing.expectEqual(Category.letter, category('a'));
    try std.testing.expectEqual(Category.letter, category('Z'));
    try std.testing.expectEqual(Category.number, category('0'));
    try std.testing.expectEqual(Category.number, category('9'));
    try std.testing.expectEqual(Category.space, category(' '));
    try std.testing.expectEqual(Category.space, category('\t'));
    try std.testing.expectEqual(Category.other, category('!'));
    try std.testing.expectEqual(Category.other, category(','));
}

test "category: common scripts" {
    try std.testing.expectEqual(Category.letter, category(0x4E2D)); // CJK 中
    try std.testing.expectEqual(Category.letter, category(0x044F)); // Cyrillic я
    try std.testing.expectEqual(Category.letter, category(0x03B1)); // Greek α
    try std.testing.expectEqual(Category.letter, category(0x3042)); // Hiragana あ
    try std.testing.expectEqual(Category.letter, category(0x30AB)); // Katakana カ
}

test "category: latin extended" {
    try std.testing.expectEqual(Category.letter, category(0x00E9)); // é
    try std.testing.expectEqual(Category.letter, category(0x00DF)); // ß
    try std.testing.expectEqual(Category.letter, category(0x00FC)); // ü
}
