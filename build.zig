const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const is_release = optimize != .Debug;
    const strip: ?bool = if (is_release) true else null;
    const single_threaded: ?bool = if (is_release) true else null;
    const omit_frame_pointer: ?bool = if (is_release) true else null;
    const error_tracing: ?bool = if (optimize == .ReleaseSmall) false else null;

    const mod = b.addModule("vantablack", .{
        .root_source_file = b.path("src/vantablack.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .single_threaded = single_threaded,
        .omit_frame_pointer = omit_frame_pointer,
        .error_tracing = error_tracing,
    });

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
