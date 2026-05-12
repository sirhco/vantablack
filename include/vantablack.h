// vantablack — C ABI header.
//
// Pair with `libvantablack.a` (cross-compiled via
// `zig build -Dtarget=aarch64-ios`). Swift hosts: drop both files into
// the Xcode project, link Foundation + Metal + QuartzCore, then use a
// bridging header that #includes this file.
//
// Stability: ABI is unstable until 1.0. Symbols may be renamed or
// re-ordered. Pin to a specific commit.

#ifndef VANTABLACK_H
#define VANTABLACK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Status codes
// ---------------------------------------------------------------------------

#define VTB_OK                  0
#define VTB_ERR_INVALID_ARG    -1
#define VTB_ERR_FILE_OPEN      -2
#define VTB_ERR_PARSE          -3
#define VTB_ERR_OOM            -4
#define VTB_ERR_METAL          -5
#define VTB_ERR_DISPATCH       -6
#define VTB_ERR_STREAM_ABORTED -7
#define VTB_ERR_UNSUPPORTED    -8

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

typedef struct VtbModel VtbModel;
typedef struct VtbState VtbState;

// ---------------------------------------------------------------------------
// Sampler config (POD)
// ---------------------------------------------------------------------------

typedef struct VtbSamplerConfig {
    float    temperature;   // 0.0 = greedy argmax
    size_t   top_k;         // 0 = disabled
    float    top_p;         // 0.0 or 1.0 = disabled
    uint64_t seed;          // PRNG seed
} VtbSamplerConfig;

// ---------------------------------------------------------------------------
// Streaming callback
// ---------------------------------------------------------------------------

// Called once per generated token. Return 0 to stop generation, non-zero
// to continue. `piece_bytes` is UTF-8; not null-terminated. `is_final`
// is 1 on the last call (EOS hit or max_tokens reached), 0 otherwise.
typedef uint8_t (*VtbTokenCb)(void *ctx,
                              uint32_t token_id,
                              const uint8_t *piece_bytes,
                              size_t piece_len,
                              uint8_t is_final);

// ---------------------------------------------------------------------------
// Model lifecycle
// ---------------------------------------------------------------------------

// Open a GGUF file. `path` must be an absolute path (hosts know their
// app's bundle root — resolution is the host's job). `metal_enabled`
// requests the GPU backend; falls back to CPU if the library was built
// without `-Dmetal=true` or device init fails. Returns null on failure.
VtbModel *vtb_model_open(const char *path, uint8_t metal_enabled);

// Free everything associated with `m`. Safe to pass null.
void vtb_model_close(VtbModel *m);

// ---------------------------------------------------------------------------
// State lifecycle
// ---------------------------------------------------------------------------

// Initialize a per-session state (KV cache + sampler). The state holds
// a non-owning reference to `m` — keep `m` alive until after
// `vtb_state_deinit`. Returns null on failure.
VtbState *vtb_state_init(VtbModel *m, const VtbSamplerConfig *cfg);

// Free per-session state. Safe to pass null.
void vtb_state_deinit(VtbState *s);

// ---------------------------------------------------------------------------
// Streaming generation
// ---------------------------------------------------------------------------

// Generate up to `max_tokens` tokens, invoking `cb` for each. Returns
// `VTB_OK` on success, negative on error (see status codes above).
int vtb_generate_stream(VtbState *s,
                        const uint8_t *prompt_bytes,
                        size_t prompt_len,
                        size_t max_tokens,
                        VtbTokenCb cb,
                        void *cb_ctx);

// ---------------------------------------------------------------------------
// Pressure / thermal signals (forward host OS notifications)
// ---------------------------------------------------------------------------

// `level`: 0=normal, 1=warning, 2=critical. Out-of-range → critical.
// Safe to call from any thread. Maps from
// UIApplicationDidReceiveMemoryWarningNotification.
void vtb_signal_memory(uint8_t level);

// `state`: 0=nominal, 1=fair, 2=serious, 3=critical. Out-of-range →
// critical. Maps from ProcessInfo.ThermalState raw value.
void vtb_signal_thermal(uint8_t state);

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

// Returns a static, null-terminated version string. Do not free.
const char *vtb_version_string(void);

// 1 if the library was built with Metal support, 0 otherwise.
uint8_t vtb_has_metal(void);

#ifdef __cplusplus
} // extern "C"
#endif
#endif // VANTABLACK_H
