//! Chat-template formatting for Llama-style chat models.
//!
//! Currently only the TinyLlama-Chat zephyr template is implemented. Expand
//! to a registry (per architecture / per model) when adding non-zephyr models.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Role = enum { system, user, assistant };

pub const Message = struct {
    role: Role,
    content: []const u8,
};

/// TinyLlama-Chat (zephyr) template:
///   <|system|>\n{system}</s>\n<|user|>\n{user}</s>\n<|assistant|>\n
///
/// Single-turn convenience wrapper.
pub fn formatLlamaChatSingle(
    arena: Allocator,
    system: []const u8,
    user: []const u8,
) ![]const u8 {
    return std.fmt.allocPrint(
        arena,
        "<|system|>\n{s}</s>\n<|user|>\n{s}</s>\n<|assistant|>\n",
        .{ system, user },
    );
}

/// Multi-turn variant. Emits one `<|{role}|>\n{content}</s>\n` block per
/// message, then closes with `<|assistant|>\n` to prompt completion.
/// If the message list omits a system message, no system block is emitted
/// (the model uses its trained default behavior).
pub fn formatLlamaChat(
    arena: Allocator,
    messages: []const Message,
) ![]const u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(arena);
    for (messages) |m| {
        try buf.print(arena, "<|{s}|>\n{s}</s>\n", .{ roleTag(m.role), m.content });
    }
    try buf.appendSlice(arena, "<|assistant|>\n");
    return buf.toOwnedSlice(arena);
}

fn roleTag(r: Role) []const u8 {
    return switch (r) {
        .system => "system",
        .user => "user",
        .assistant => "assistant",
    };
}

test "single-turn template shape" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const out = try formatLlamaChatSingle(arena.allocator(), "be helpful", "hi");
    try std.testing.expectEqualStrings(
        "<|system|>\nbe helpful</s>\n<|user|>\nhi</s>\n<|assistant|>\n",
        out,
    );
}

test "multi-turn template appends assistant prompt" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const msgs = [_]Message{
        .{ .role = .system, .content = "be helpful" },
        .{ .role = .user, .content = "first" },
        .{ .role = .assistant, .content = "ok" },
        .{ .role = .user, .content = "second" },
    };
    const out = try formatLlamaChat(arena.allocator(), &msgs);
    try std.testing.expectEqualStrings(
        "<|system|>\nbe helpful</s>\n<|user|>\nfirst</s>\n<|assistant|>\nok</s>\n<|user|>\nsecond</s>\n<|assistant|>\n",
        out,
    );
}
