//! Streaming generation.
//!
//! `generateStream` runs the prompt-prefill + sampling loop and invokes a
//! caller-provided callback for each generated token. The callback can
//! return `false` to terminate generation early — the typical mobile-chat
//! UI pattern of "stop on user cancel" lands without polling.
//!
//! API shape favours C ABI consumers (Swift on iOS, Kotlin on Android via
//! JNI later): the callback takes pointer+length rather than a Zig slice,
//! and a u8 boolean rather than `bool`.

const std = @import("std");

const forward_mod = @import("forward.zig");
const kv_cache_mod = @import("kv_cache.zig");
const model_mod = @import("model.zig");
const sampler_mod = @import("sampler.zig");
const tokenizer_mod = @import("tokenizer.zig");
const backend_mod = @import("backend.zig");

const Backend = backend_mod.Backend;
const KvCache = kv_cache_mod.KvCache;
const Model = model_mod.Model;
const Sampler = sampler_mod.Sampler;
const State = forward_mod.State;
const Tokenizer = tokenizer_mod.Tokenizer;

/// C-ABI-safe token callback. Returns 0 to stop, non-zero to continue.
/// `is_final` is true on the last call of a generation run (EOS hit or
/// `max_tokens` reached).
pub const TokenCallback = *const fn (
    ctx: *anyopaque,
    token_id: u32,
    piece_bytes: [*]const u8,
    piece_len: usize,
    is_final: u8,
) callconv(.c) u8;

pub const StreamError = error{
    OutOfMemory,
    DecodeBufferTooSmall,
    TokenOutOfRange,
    KvCacheFull,
    UnsupportedWeightType,
};

/// Single-line decode buffer. 256 bytes is enough for any single piece in
/// SentencePiece / byte-level BPE vocabularies in use today; longer
/// callbacks should compose multi-token spans on their own side.
const PieceBufBytes: usize = 256;

/// Streaming prompt-and-generate. Encodes the prompt with the supplied
/// tokenizer, replays it through the backend, then samples up to
/// `max_tokens` new tokens. The callback fires once per generated token.
///
/// Caller owns `model`, `tokenizer`, `sampler`, `backend`, `cache`. The
/// kv-cache must be empty or this picks up where the previous call left
/// off (the natural chat-turn append model).
pub fn generateStream(
    allocator: std.mem.Allocator,
    model: *const Model,
    tokenizer: *const Tokenizer,
    sampler: *Sampler,
    backend: Backend,
    state: *State,
    cache: *KvCache,
    prompt: []const u8,
    max_tokens: usize,
    callback: TokenCallback,
    cb_ctx: *anyopaque,
) StreamError!void {
    // Prefill: encode prompt tokens, replay through the backend.
    const prompt_ids = tokenizer.encode(allocator, prompt, true) catch return error.OutOfMemory;
    defer allocator.free(prompt_ids);

    for (prompt_ids) |id| {
        forward_mod.stepWithBackend(model, state, cache, backend, id) catch |err| return mapErr(err);
    }
    if (max_tokens == 0) return;

    var produced: usize = 0;
    var next: u32 = sampler.sample(state.logits);

    while (produced < max_tokens) : (produced += 1) {
        const eos_hit = next == tokenizer.eos;
        const is_last = eos_hit or produced + 1 == max_tokens;

        var piece_buf: [PieceBufBytes]u8 = undefined;
        const piece = renderPiece(tokenizer, next, &piece_buf) catch
            return error.DecodeBufferTooSmall;

        const cont = callback(cb_ctx, next, piece.ptr, piece.len, @intFromBool(is_last));
        if (cont == 0 or eos_hit) return;

        forward_mod.stepWithBackend(model, state, cache, backend, next) catch |err| return mapErr(err);
        next = sampler.sample(state.logits);
    }
}

fn renderPiece(tokenizer: *const Tokenizer, token_id: u32, buf: []u8) ![]u8 {
    // Fixed-buffer Writer: the decode cannot allocate. An oversize piece
    // (would only happen on a corrupt vocab) returns an error rather than
    // silently truncating.
    var fixed: std.Io.Writer = .fixed(buf);
    try tokenizer.decodeTo(&fixed, token_id);
    return buf[0..fixed.end];
}

fn mapErr(err: anyerror) StreamError {
    return switch (err) {
        error.TokenOutOfRange => error.TokenOutOfRange,
        error.KvCacheFull => error.KvCacheFull,
        error.UnsupportedWeightType => error.UnsupportedWeightType,
        error.OutOfMemory => error.OutOfMemory,
        else => error.UnsupportedWeightType,
    };
}

// -- tests ----------------------------------------------------------------

const TestState = struct {
    called: usize = 0,
    last_token: u32 = 0,
    last_final: u8 = 0,
    stop_after: usize = std.math.maxInt(usize),
    seen_tokens: [16]u32 = .{0} ** 16,
    seen_pieces: [16][PieceBufBytes]u8 = undefined,
    seen_piece_lens: [16]usize = .{0} ** 16,
};

fn captureCb(
    ctx_raw: *anyopaque,
    token_id: u32,
    piece_bytes: [*]const u8,
    piece_len: usize,
    is_final: u8,
) callconv(.c) u8 {
    const ts: *TestState = @ptrCast(@alignCast(ctx_raw));
    ts.last_token = token_id;
    ts.last_final = is_final;
    if (ts.called < ts.seen_tokens.len) {
        ts.seen_tokens[ts.called] = token_id;
        const copy_len = @min(piece_len, PieceBufBytes);
        @memcpy(ts.seen_pieces[ts.called][0..copy_len], piece_bytes[0..copy_len]);
        ts.seen_piece_lens[ts.called] = copy_len;
    }
    ts.called += 1;
    if (ts.called > ts.stop_after) return 0;
    return 1;
}

test "TokenCallback signature is C-ABI compatible" {
    // Compile-time check: the function pointer can be assigned and called.
    const cb: TokenCallback = captureCb;
    var ts = TestState{};
    var dummy_bytes = [_]u8{ 'a', 'b' };
    const rc = cb(@ptrCast(&ts), 42, &dummy_bytes, 2, 1);
    try std.testing.expectEqual(@as(u8, 1), rc);
    try std.testing.expectEqual(@as(u32, 42), ts.last_token);
    try std.testing.expectEqual(@as(u8, 1), ts.last_final);
    try std.testing.expectEqual(@as(usize, 1), ts.called);
}

test "TokenCallback early-stop signal terminates" {
    const cb: TokenCallback = captureCb;
    var ts = TestState{ .stop_after = 1 };
    var dummy_bytes = [_]u8{ 'x' };
    _ = cb(@ptrCast(&ts), 1, &dummy_bytes, 1, 0);
    const rc = cb(@ptrCast(&ts), 2, &dummy_bytes, 1, 0);
    try std.testing.expectEqual(@as(u8, 0), rc);
}
