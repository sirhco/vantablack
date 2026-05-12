const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const target_os = target.result.os.tag;
    const is_ios = target_os == .ios;

    const is_release = optimize != .Debug;
    const strip: ?bool = if (is_release) true else null;
    // ReleaseSmall stays single-threaded for binary size; ReleaseFast/Safe
    // get the thread pool for matmul parallelism.
    const single_threaded: ?bool = if (optimize == .ReleaseSmall) true else null;
    const omit_frame_pointer: ?bool = if (is_release) true else null;
    const error_tracing: ?bool = if (optimize == .ReleaseSmall) false else null;

    // Apple Metal GPU backend. Off by default on macOS (preserves pure-Zig
    // path + ReleaseSmall binary size); ON by default on iOS (every iOS
    // target has a Metal GPU and consumers want hardware acceleration —
    // no CPU-only fallback ships in the iOS library).
    const metal_default = is_ios;
    const metal = b.option(bool, "metal", "build with Apple Metal GPU backend") orelse metal_default;
    // Pre-concatenate Q/K/V and gate/up weights at Model.init so each fused
    // matmul reads activations once. Adds ~660 MB residency for TinyLlama
    // Q8_0 (~2x for the fused groups). Off by default.
    const weight_fusion = b.option(bool, "weight_fusion", "pre-concatenate Q/K/V + gate/up weights at init") orelse false;
    // HTTP server (Ollama-compatible). On by default on desktop; OFF by
    // default on iOS (the static archive should not pull in std.http).
    const server_default = !is_ios;
    const server = b.option(bool, "server", "include HTTP server (Ollama API)") orelse server_default;

    const build_options = b.addOptions();
    build_options.addOption(bool, "metal", metal);
    build_options.addOption(bool, "weight_fusion", weight_fusion);
    build_options.addOption(bool, "server", server);

    const mod = b.addModule("vantablack", .{
        .root_source_file = b.path("src/vantablack.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .single_threaded = single_threaded,
        .omit_frame_pointer = omit_frame_pointer,
        .error_tracing = error_tracing,
    });
    mod.addOptions("build_options", build_options);
    if (metal) {
        attachMetal(b, mod, target_os);
    }

    // iOS builds skip the CLI executable and the HTTP server. The
    // deliverable is a static archive consumed by a Swift wrapper. Strip
    // is force-disabled for iOS: a stripped static archive would lose
    // the C-ABI `export fn` symbols the consumer links against.
    if (is_ios) {
        // Re-create the module with strip=false (the original `mod` may
        // have been built with strip=true on release optimizes).
        const lib_mod = b.addModule("vantablack-ios", .{
            .root_source_file = b.path("src/vantablack.zig"),
            .target = target,
            .optimize = optimize,
            .strip = false,
            .single_threaded = single_threaded,
            .omit_frame_pointer = omit_frame_pointer,
            .error_tracing = error_tracing,
        });
        lib_mod.addOptions("build_options", build_options);
        if (metal) attachMetal(b, lib_mod, target_os);
        const lib = b.addLibrary(.{
            .name = "vantablack",
            .root_module = lib_mod,
            .linkage = .static,
        });
        b.installArtifact(lib);
        // Drop the C header next to the static archive so a Swift host
        // can `#include "vantablack.h"` from its bridging header.
        b.installFile("include/vantablack.h", "include/vantablack.h");

        // Provide a `test` step that runs the library tests on the host
        // (iOS cannot run tests cross-compile; the library tests build
        // for host but the artifact is iOS).
        const mod_tests = b.addTest(.{ .root_module = lib_mod });
        const run_mod_tests = b.addRunArtifact(mod_tests);
        const test_step = b.step("test", "Run tests");
        test_step.dependOn(&run_mod_tests.step);
        return;
    }

    const exe = b.addExecutable(.{
        .name = "vantablack",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .strip = strip,
            .single_threaded = single_threaded,
            .omit_frame_pointer = omit_frame_pointer,
            .error_tracing = error_tracing,
            .imports = &.{
                .{ .name = "vantablack", .module = mod },
            },
        }),
    });
    exe.root_module.addOptions("build_options", build_options);

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const mod_tests = b.addTest(.{ .root_module = mod });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{ .root_module = exe.root_module });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
}

fn attachMetal(b: *std.Build, mod: *std.Build.Module, target_os: std.Target.Os.Tag) void {
    mod.link_libc = true; // Obj-C runtime needs libc

    // iOS cross-compile from a macOS host needs the Xcode SDK path so
    // clang can find Foundation.framework / Metal.framework. We invoke
    // `xcrun` once at build configuration time and feed the result back
    // into the module's framework + header search paths. Skipped when
    // running on macOS native — those frameworks come from the active
    // CommandLineTools install via Zig's default search path.
    if (target_os == .ios) {
        const sdk_path = resolveIosSdk(b) catch |err| {
            std.debug.print("vantablack: failed to resolve iOS SDK via xcrun ({}); install Xcode + run `xcode-select --install`\n", .{err});
            @panic("iOS SDK unavailable");
        };
        const frameworks_path = b.fmt("{s}/System/Library/Frameworks", .{sdk_path});
        const include_path = b.fmt("{s}/usr/include", .{sdk_path});
        const lib_path = b.fmt("{s}/usr/lib", .{sdk_path});
        mod.addSystemFrameworkPath(.{ .cwd_relative = frameworks_path });
        mod.addIncludePath(.{ .cwd_relative = include_path });
        mod.addLibraryPath(.{ .cwd_relative = lib_path });
    }

    mod.linkFramework("Foundation", .{});
    mod.linkFramework("Metal", .{});
    // CoreGraphics is macOS-only. On iOS, QuartzCore covers the analogous
    // CALayer / display-link symbol surface. bridge.m doesn't currently
    // touch either, but linking the right framework keeps a future
    // bridge expansion from failing at link time.
    if (target_os == .ios) {
        mod.linkFramework("QuartzCore", .{});
    } else {
        mod.linkFramework("CoreGraphics", .{});
    }
    mod.addCSourceFile(.{
        .file = b.path("src/metal/bridge.m"),
        .flags = &.{
            "-fobjc-arc",
            "-Wall",
            "-Wextra",
        },
        .language = .objective_c,
    });
}

/// Runs `xcrun --sdk iphoneos --show-sdk-path` at build-configuration
/// time. Builds on macOS with Xcode installed; fails the build with a
/// helpful message otherwise (Linux hosts can't cross-compile to iOS
/// without a copy of the SDK).
fn resolveIosSdk(b: *std.Build) ![]const u8 {
    const stdout = b.run(&.{ "xcrun", "--sdk", "iphoneos", "--show-sdk-path" });
    // Trim trailing newline.
    var end = stdout.len;
    while (end > 0 and (stdout[end - 1] == '\n' or stdout[end - 1] == '\r')) end -= 1;
    return stdout[0..end];
}
