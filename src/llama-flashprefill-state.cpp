// llama-flashprefill-state.cpp — pure versioned policy envelope codec.
//
// No llama.h / llama-io.h / KV / GPU dependency; the only include beyond the
// C++ standard library is the public frozen policy header for the schema
// version constants. All functions are pure and deterministic.
// Host byte order (memcpy field order) matches llama_io raw read/write, which
// is already host-order for every other state field.

#include "llama-flashprefill-state.h"

#include <cstdio>
#include <cstring>
#include <limits>

namespace llama_flashprefill_state {

namespace {

void store_u32(uint8_t * dst, uint32_t v) { std::memcpy(dst, &v, sizeof(v)); }
void store_u64(uint8_t * dst, uint64_t v) { std::memcpy(dst, &v, sizeof(v)); }
uint32_t load_u32(const uint8_t * src) { uint32_t v = 0; std::memcpy(&v, src, sizeof(v)); return v; }
uint64_t load_u64(const uint8_t * src) { uint64_t v = 0; std::memcpy(&v, src, sizeof(v)); return v; }

void set_error(std::string * error, const char * msg) {
    if (error != nullptr) {
        *error = msg;
    }
}

} // namespace

envelope make_envelope(
        uint32_t scope, uint64_t policy_fp,
        uint64_t proc_nonce0, uint64_t proc_nonce1,
        uint64_t context_serial, uint64_t adapter_gen) {
    envelope env;
    env.magic          = kMagic;
    env.version        = kVersion;
    env.scope          = scope;
    env.fp_version     = LLAMA_FLASHPREFILL_VERSION;
    env.config_version = LLAMA_FLASHPREFILL_CONFIG_VERSION;
    env.reserved       = 0;
    env.policy_fp      = policy_fp;
    env.proc_nonce0    = proc_nonce0;
    env.proc_nonce1    = proc_nonce1;
    env.context_serial = context_serial;
    env.adapter_gen    = adapter_gen;
    return env;
}

bool encode_envelope(const envelope & env, uint8_t * dst, size_t dst_size, size_t * out_n) {
    if (dst == nullptr || dst_size < kEnvelopeBytes) {
        return false;
    }
    // Fail closed at encode time with the SAME validity bar as decode+match:
    // never persist a header that load would have to reject. Covers unknown
    // scope, schema skew, zero identity fields, and saturated (MAX) adapter
    // generation — a fresh save is never MAX (match() rejects MAX either way).
    if (env.magic != kMagic || env.version != kVersion || env.reserved != 0) {
        return false;
    }
    if (env.scope != kScopeFull && env.scope != kScopeSeq) {
        return false;
    }
    if (env.fp_version != LLAMA_FLASHPREFILL_VERSION || env.config_version != LLAMA_FLASHPREFILL_CONFIG_VERSION) {
        return false;
    }
    if (env.policy_fp == 0 || env.proc_nonce0 == 0 || env.proc_nonce1 == 0 ||
        env.context_serial == 0 || env.adapter_gen == 0) {
        return false;
    }
    if (env.adapter_gen == std::numeric_limits<uint64_t>::max()) {
        return false;
    }
    size_t o = 0;
    store_u32(dst + o, env.magic);            o += 4;
    store_u32(dst + o, env.version);          o += 4;
    store_u32(dst + o, env.scope);            o += 4;
    store_u32(dst + o, env.fp_version);       o += 4;
    store_u32(dst + o, env.config_version);   o += 4;
    store_u32(dst + o, env.reserved);         o += 4;
    store_u64(dst + o, env.policy_fp);        o += 8;
    store_u64(dst + o, env.proc_nonce0);      o += 8;
    store_u64(dst + o, env.proc_nonce1);      o += 8;
    store_u64(dst + o, env.context_serial);   o += 8;
    store_u64(dst + o, env.adapter_gen);      o += 8;
    if (out_n != nullptr) {
        *out_n = o;
    }
    return o == kEnvelopeBytes;
}

bool decode_envelope(const uint8_t * src, size_t src_size, envelope * out, size_t * out_n, std::string * error) {
    if (src == nullptr || out == nullptr || src_size < kEnvelopeBytes) {
        set_error(error, "flashprefill state: missing policy header (legacy bytes without an envelope); re-prefill under the current policy instead of reusing this state");
        return false;
    }
    envelope env;
    size_t o = 0;
    env.magic            = load_u32(src + o); o += 4;
    env.version          = load_u32(src + o); o += 4;
    env.scope            = load_u32(src + o); o += 4;
    env.fp_version       = load_u32(src + o); o += 4;
    env.config_version   = load_u32(src + o); o += 4;
    env.reserved         = load_u32(src + o); o += 4;
    env.policy_fp        = load_u64(src + o); o += 8;
    env.proc_nonce0      = load_u64(src + o); o += 8;
    env.proc_nonce1      = load_u64(src + o); o += 8;
    env.context_serial   = load_u64(src + o); o += 8;
    env.adapter_gen      = load_u64(src + o); o += 8;

    if (env.magic != kMagic) {
        set_error(error, "flashprefill state: missing policy header (unrecognized magic — legacy OFF bytes or foreign state); re-prefill under the current policy instead of reusing this state");
        return false;
    }
    if (env.version != kVersion) {
        set_error(error, "flashprefill state: unknown envelope version; upgrade the binary and re-prefill instead of reusing this state");
        return false;
    }
    if (env.scope != kScopeFull && env.scope != kScopeSeq) {
        set_error(error, "flashprefill state: unknown envelope scope; re-prefill under the current policy instead of reusing this state");
        return false;
    }
    if (env.fp_version != LLAMA_FLASHPREFILL_VERSION || env.config_version != LLAMA_FLASHPREFILL_CONFIG_VERSION) {
        set_error(error, "flashprefill state: policy schema mismatch (algorithm/config version skew); re-prefill under the current policy instead of reusing this state");
        return false;
    }
    if (env.reserved != 0) {
        set_error(error, "flashprefill state: corrupt envelope flags; re-prefill under the current policy instead of reusing this state");
        return false;
    }
    if (env.policy_fp == 0 || env.proc_nonce0 == 0 || env.proc_nonce1 == 0 ||
        env.context_serial == 0 || env.adapter_gen == 0) {
        set_error(error, "flashprefill state: corrupt envelope identity (zero fingerprint/nonce/serial/generation); re-prefill under the current policy instead of reusing this state");
        return false;
    }
    *out = env;
    if (out_n != nullptr) {
        *out_n = o;
    }
    return true;
}

bool encode_framed_envelope(const envelope & env, uint8_t * dst, size_t dst_size, size_t * out_n) {
    if (dst == nullptr || dst_size < kFramedBytes) {
        return false;
    }
    // Body first: encode_envelope validates fully before storing anything, so
    // a failing body leaves dst untouched for the caller to discard.
    uint8_t * const body = dst + 4u + kFrameMarker.size();
    if (!encode_envelope(env, body, kEnvelopeBytes)) {
        return false;
    }
    store_u32(dst, (uint32_t) kFrameMarker.size());
    std::memcpy(body - kFrameMarker.size(), kFrameMarker.data(), kFrameMarker.size());
    if (out_n != nullptr) {
        *out_n = kFramedBytes;
    }
    return true;
}

bool decode_framed_envelope(const uint8_t * src, size_t src_size, envelope * out, size_t * out_n, std::string * error) {
    if (src == nullptr || out == nullptr || src_size < kFramedBytes) {
        set_error(error, "flashprefill state: missing policy header (short bytes without a framed envelope — legacy OFF state or truncation); re-prefill under the current policy instead of reusing this state");
        return false;
    }
    // Length word first, before touching anything else: legacy arch lengths,
    // foreign magics (including the unreleased raw-64B magic, which reads as
    // ~1.18 GB), and short/corrupt prefixes all fail here with no further
    // reads and no allocation.
    if (load_u32(src) != (uint32_t) kFrameMarker.size()) {
        set_error(error, "flashprefill state: missing policy header (no framing marker — legacy OFF bytes or foreign state); re-prefill under the current policy instead of reusing this state");
        return false;
    }
    if (std::memcmp(src + 4u, kFrameMarker.data(), kFrameMarker.size()) != 0) {
        set_error(error, "flashprefill state: missing policy header (framing marker mismatch — foreign state); re-prefill under the current policy instead of reusing this state");
        return false;
    }
    size_t body_n = 0;
    if (!decode_envelope(src + 4u + kFrameMarker.size(), kEnvelopeBytes, out, &body_n, error)) {
        return false;
    }
    if (out_n != nullptr) {
        *out_n = kFramedBytes;
    }
    return body_n == kEnvelopeBytes;
}

bool match_envelope(
        const envelope & stored,
        uint32_t         expected_scope,
        uint64_t         expected_policy_fp,
        uint64_t         expected_nonce0,
        uint64_t         expected_nonce1,
        uint64_t         expected_serial,
        uint64_t         expected_adapter_gen,
        std::string *    error) {
    if (stored.scope != expected_scope) {
        set_error(error, expected_scope == kScopeFull
            ? "flashprefill state: scope mismatch (per-sequence state cannot restore a full context, and vice versa); re-prefill under the current policy instead of reusing this state"
            : "flashprefill state: scope mismatch (full-context state cannot restore a single sequence, and vice versa); re-prefill under the current policy instead of reusing this state");
        return false;
    }
    if (stored.fp_version != LLAMA_FLASHPREFILL_VERSION || stored.config_version != LLAMA_FLASHPREFILL_CONFIG_VERSION) {
        set_error(error, "flashprefill state: policy schema mismatch (algorithm/config version skew); re-prefill under the current policy instead of reusing this state");
        return false;
    }
    if (stored.policy_fp != expected_policy_fp) {
        set_error(error, "flashprefill state: policy fingerprint mismatch (different approximation config, model identity, or tail/packing policy); re-prefill under the current policy instead of reusing this state");
        return false;
    }
    // Exact process-nonce compare BEFORE the serial: a bare counter restarts
    // at 1 in every process, so the counter alone can never authorize a
    // restore. A nonce mismatch means a different process (restart or another
    // instance) even when serial/policy/adapter all coincide.
    if (stored.proc_nonce0 != expected_nonce0 || stored.proc_nonce1 != expected_nonce1) {
        set_error(error, "flashprefill state: process nonce mismatch (state belongs to a different process lifetime — restart or another instance; the serial counter alone is not cross-process identity); re-prefill in this process instead of reusing this state");
        return false;
    }
    if (stored.context_serial != expected_serial) {
        set_error(error, "flashprefill state: context serial mismatch (state belongs to a different live context; model.desc is not a weight hash, so cross-context durable reuse is rejected); re-prefill in this context instead of reusing this state");
        return false;
    }
    constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
    if (stored.adapter_gen == kMax || expected_adapter_gen == kMax) {
        set_error(error, "flashprefill state: adapter generation saturated (always-invalid, like CellGeneration MAX); re-prefill under the current adapters instead of reusing this state");
        return false;
    }
    if (stored.adapter_gen != expected_adapter_gen) {
        set_error(error, "flashprefill state: adapter generation mismatch (LoRA set/remove/scale or cvec mutation since save); re-prefill under the current adapters instead of reusing this state");
        return false;
    }
    return true;
}

uint64_t state_cache_key(uint64_t policy_fp, uint64_t context_serial, uint64_t adapter_gen) {
    // FNV-1a 64 over a domain separator plus the three identity inputs, so
    // the key domain cannot collide with the bare policy fingerprint domain.
    // Same-process fast-path stamp only: the process nonce is constant within
    // a process, so it adds no discrimination here and is intentionally NOT
    // mixed in (the envelope, which carries the nonce, stays authoritative).
    uint64_t h = 14695981039346656037ull;
    const auto mix_u64 = [&](uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            h ^= (uint64_t) ((v >> (8 * i)) & 0xffu);
            h *= 1099511628211ull;
        }
    };
    mix_u64(0x4650535441544531ull); // 'FPSTATE1' domain separator
    mix_u64(policy_fp);
    mix_u64(context_serial);
    mix_u64(adapter_gen);
    return h == 0 ? 1 : h;
}

std::string describe_envelope(const envelope & env) {
    char buf[192];
    const char * scope = env.scope == kScopeFull ? "full" : (env.scope == kScopeSeq ? "seq" : "?");
    std::snprintf(buf, sizeof(buf), "FPS0 v%u scope=%s policy=0x%016llx nonce=0x%016llx%016llx serial=0x%016llx adapter=%llu",
        (unsigned) env.version, scope,
        (unsigned long long) env.policy_fp,
        (unsigned long long) env.proc_nonce0,
        (unsigned long long) env.proc_nonce1,
        (unsigned long long) env.context_serial,
        (unsigned long long) env.adapter_gen);
    return std::string(buf);
}

} // namespace llama_flashprefill_state
