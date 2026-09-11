// test-flashprefill-state-envelope.h — FlashPrefill V2 state envelope unit coverage.
//
// Pure helpers from src/llama-flashprefill-state.h
// (namespace llama_flashprefill_state) only: no model, no KV, no backend, no
// GPU, no threads. Every CHECK below distinguishes a real state-isolation
// failure mode (alien policy reuse, cross-process/restart reuse, cross-context
// restore, stale-adapter reuse, scope confusion, legacy bytes trusted as
// versioned state).
//
// Build note: this header intentionally defines NO main, so a larger harness
// (e.g. tests/test-flashprefill-state.cpp) can #include it and call
// flashprefill_state_envelope_tests::run_tests() without a duplicate-main
// link. The standalone driver is the sibling
// test-flashprefill-state-envelope.cpp (registered separately; this header
// plus the src/llama-flashprefill-state.cpp link already in src/CMakeLists.txt
// are all it needs).

#pragma once

#include "../src/llama-flashprefill-state.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace flashprefill_state_envelope_tests {

namespace fpst = llama_flashprefill_state;

static int s_failures = 0;

#define FPENV_CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++s_failures; \
    } \
} while (0)

// Fixed nonzero identities used across cases (never 0/MAX: those are the
// always-invalid domains, covered explicitly below).
static const uint64_t kPolicy = 0x123456789abcdef0ull;
static const uint64_t kNonce0 = 0x1111222233334444ull;
static const uint64_t kNonce1 = 0x5555666677778888ull;
static const uint64_t kSerial = 0x0fedcba987654321ull;
static const uint64_t kAdapt  = 7u;

static void test_round_trip_full_and_seq() {
    for (uint32_t scope : {fpst::kScopeFull, fpst::kScopeSeq}) {
        const fpst::envelope env = fpst::make_envelope(scope, kPolicy, kNonce0, kNonce1, kSerial, kAdapt);
        uint8_t buf[fpst::kEnvelopeBytes];
        std::memset(buf, 0xcc, sizeof(buf));
        size_t n = 0;
        FPENV_CHECK(fpst::encode_envelope(env, buf, sizeof(buf), &n));
        FPENV_CHECK(n == fpst::envelope_nbytes());
        FPENV_CHECK(fpst::envelope_nbytes() == 64u);
        fpst::envelope back;
        std::string error;
        size_t m = 0;
        FPENV_CHECK(fpst::decode_envelope(buf, sizeof(buf), &back, &m, &error));
        FPENV_CHECK(m == sizeof(buf));
        FPENV_CHECK(back.magic == fpst::kMagic);
        FPENV_CHECK(back.version == fpst::kVersion);
        FPENV_CHECK(back.scope == scope);
        FPENV_CHECK(back.policy_fp == kPolicy);
        FPENV_CHECK(back.proc_nonce0 == kNonce0);
        FPENV_CHECK(back.proc_nonce1 == kNonce1);
        FPENV_CHECK(back.context_serial == kSerial);
        FPENV_CHECK(back.adapter_gen == kAdapt);
        // Exact match against the live identity validates.
        FPENV_CHECK(fpst::match_envelope(back, scope, kPolicy, kNonce0, kNonce1, kSerial, kAdapt, &error));
    }
}

static void test_rejects_legacy_and_corrupt() {
    // Legacy OFF bytes (an arch-string prefix: u32 len + "gpt2...") must
    // decode as a MISSING header, never as a valid envelope.
    uint8_t legacy[fpst::kEnvelopeBytes];
    std::memset(legacy, 0, sizeof(legacy));
    const uint32_t arch_len = 4;
    std::memcpy(legacy, &arch_len, sizeof(arch_len));
    std::memcpy(legacy + 4, "gpt2", 4);
    {
        fpst::envelope out;
        std::string error;
        FPENV_CHECK(!fpst::decode_envelope(legacy, sizeof(legacy), &out, nullptr, &error));
        FPENV_CHECK(!error.empty());
    }
    // A 48-byte prefix of a valid envelope (old width) is truncated, not valid.
    {
        const fpst::envelope good = fpst::make_envelope(fpst::kScopeFull, kPolicy, kNonce0, kNonce1, kSerial, kAdapt);
        uint8_t buf[fpst::kEnvelopeBytes];
        FPENV_CHECK(fpst::encode_envelope(good, buf, sizeof(buf)));
        fpst::envelope out;
        std::string error;
        FPENV_CHECK(!fpst::decode_envelope(buf, 48u, &out, nullptr, &error));
        uint8_t tiny[8] = {0};
        FPENV_CHECK(!fpst::decode_envelope(tiny, sizeof(tiny), &out, nullptr, &error));
        FPENV_CHECK(!fpst::decode_envelope(nullptr, 0, &out, nullptr, &error));
    }
    // Unknown version / unknown scope reject.
    {
        const fpst::envelope good = fpst::make_envelope(fpst::kScopeFull, kPolicy, kNonce0, kNonce1, kSerial, kAdapt);
        uint8_t buf[fpst::kEnvelopeBytes];
        FPENV_CHECK(fpst::encode_envelope(good, buf, sizeof(buf)));
        fpst::envelope out;
        std::string error;
        uint8_t bad[fpst::kEnvelopeBytes];
        std::memcpy(bad, buf, sizeof(bad));
        // version field is the 2nd u32 (offset 4)
        const uint32_t vbad = fpst::kVersion + 1;
        std::memcpy(bad + 4, &vbad, sizeof(vbad));
        FPENV_CHECK(!fpst::decode_envelope(bad, sizeof(bad), &out, nullptr, &error));
        std::memcpy(bad, buf, sizeof(bad));
        const uint32_t sbad = 0x7fffffffu;
        std::memcpy(bad + 8, &sbad, sizeof(sbad));
        FPENV_CHECK(!fpst::decode_envelope(bad, sizeof(bad), &out, nullptr, &error));
    }
    // Zero identity fields and saturated adapter generation never encode: a
    // fresh save is never 0/MAX (fail closed at save time, same bar as load).
    {
        uint8_t buf[fpst::kEnvelopeBytes];
        FPENV_CHECK(!fpst::encode_envelope(fpst::make_envelope(fpst::kScopeFull, 0, kNonce0, kNonce1, kSerial, kAdapt), buf, sizeof(buf)));
        FPENV_CHECK(!fpst::encode_envelope(fpst::make_envelope(fpst::kScopeFull, kPolicy, 0, kNonce1, kSerial, kAdapt), buf, sizeof(buf)));
        FPENV_CHECK(!fpst::encode_envelope(fpst::make_envelope(fpst::kScopeFull, kPolicy, kNonce0, 0, kSerial, kAdapt), buf, sizeof(buf)));
        FPENV_CHECK(!fpst::encode_envelope(fpst::make_envelope(fpst::kScopeFull, kPolicy, kNonce0, kNonce1, 0, kAdapt), buf, sizeof(buf)));
        FPENV_CHECK(!fpst::encode_envelope(fpst::make_envelope(fpst::kScopeFull, kPolicy, kNonce0, kNonce1, kSerial, 0), buf, sizeof(buf)));
        constexpr uint64_t kMax = 0xffffffffffffffffull;
        FPENV_CHECK(!fpst::encode_envelope(fpst::make_envelope(fpst::kScopeFull, kPolicy, kNonce0, kNonce1, kSerial, kMax), buf, sizeof(buf)));
        FPENV_CHECK(!fpst::encode_envelope(fpst::make_envelope(99u, kPolicy, kNonce0, kNonce1, kSerial, kAdapt), buf, sizeof(buf)));
        FPENV_CHECK(!fpst::encode_envelope(fpst::make_envelope(fpst::kScopeFull, kPolicy, kNonce0, kNonce1, kSerial, kAdapt), nullptr, 0));
    }
}

static void test_match_domains() {
    const fpst::envelope full = fpst::make_envelope(fpst::kScopeFull, kPolicy, kNonce0, kNonce1, kSerial, kAdapt);
    std::string error;
    // Cross-scope reuse rejects both directions.
    FPENV_CHECK(!fpst::match_envelope(full, fpst::kScopeSeq, kPolicy, kNonce0, kNonce1, kSerial, kAdapt, &error));
    FPENV_CHECK(!error.empty());
    const fpst::envelope seq = fpst::make_envelope(fpst::kScopeSeq, kPolicy, kNonce0, kNonce1, kSerial, kAdapt);
    FPENV_CHECK(!fpst::match_envelope(seq, fpst::kScopeFull, kPolicy, kNonce0, kNonce1, kSerial, kAdapt, &error));
    // Alien policy, foreign process nonce (restart/another instance, even with
    // the same counter serial), foreign context serial, and stale adapter
    // generation each reject with a diagnostic (never silently usable).
    FPENV_CHECK(!fpst::match_envelope(full, fpst::kScopeFull, kPolicy ^ 1u, kNonce0, kNonce1, kSerial, kAdapt, &error));
    FPENV_CHECK(!error.empty());
    FPENV_CHECK(!fpst::match_envelope(full, fpst::kScopeFull, kPolicy, kNonce0 ^ 1u, kNonce1, kSerial, kAdapt, &error));
    FPENV_CHECK(!error.empty());
    FPENV_CHECK(!fpst::match_envelope(full, fpst::kScopeFull, kPolicy, kNonce0, kNonce1 ^ 1u, kSerial, kAdapt, &error));
    FPENV_CHECK(!error.empty());
    FPENV_CHECK(!fpst::match_envelope(full, fpst::kScopeFull, kPolicy, kNonce0, kNonce1, kSerial + 1, kAdapt, &error));
    FPENV_CHECK(!error.empty());
    FPENV_CHECK(!fpst::match_envelope(full, fpst::kScopeFull, kPolicy, kNonce0, kNonce1, kSerial, kAdapt + 1, &error));
    FPENV_CHECK(!error.empty());
    // Saturated (MAX) adapter generation is always-invalid on either side,
    // like CellGeneration: even an exact MAX==MAX pair must not validate.
    constexpr uint64_t kMax = 0xffffffffffffffffull;
    const fpst::envelope saturated = fpst::make_envelope(fpst::kScopeFull, kPolicy, kNonce0, kNonce1, kSerial, kMax);
    FPENV_CHECK(!fpst::match_envelope(saturated, fpst::kScopeFull, kPolicy, kNonce0, kNonce1, kSerial, kMax, &error));
    FPENV_CHECK(!fpst::match_envelope(full, fpst::kScopeFull, kPolicy, kNonce0, kNonce1, kSerial, kMax, &error));
}

static void test_cache_key_discrimination() {
    const uint64_t base = fpst::state_cache_key(kPolicy, kSerial, kAdapt);
    FPENV_CHECK(base != 0);
    // Deterministic for identical inputs; in-memory fast path only, never
    // serialized as authority.
    FPENV_CHECK(fpst::state_cache_key(kPolicy, kSerial, kAdapt) == base);
    // Any identity difference retires the key (policy, serial, adapter).
    FPENV_CHECK(fpst::state_cache_key(kPolicy ^ 1u, kSerial, kAdapt) != base);
    FPENV_CHECK(fpst::state_cache_key(kPolicy, kSerial + 1, kAdapt) != base);
    FPENV_CHECK(fpst::state_cache_key(kPolicy, kSerial, kAdapt + 1) != base);
}

static void test_framed_round_trip_and_rejects() {
    for (uint32_t scope : {fpst::kScopeFull, fpst::kScopeSeq}) {
        const fpst::envelope env = fpst::make_envelope(scope, kPolicy, kNonce0, kNonce1, kSerial, kAdapt);
        uint8_t buf[fpst::kFramedBytes];
        std::memset(buf, 0xcc, sizeof(buf));
        size_t n = 0;
        FPENV_CHECK(fpst::encode_framed_envelope(env, buf, sizeof(buf), &n));
        FPENV_CHECK(n == fpst::framed_nbytes());
        uint32_t lead_len = 0;
        std::memcpy(&lead_len, buf, sizeof(lead_len));
        FPENV_CHECK(lead_len == (uint32_t) fpst::kFrameMarker.size());
        FPENV_CHECK(std::memcmp(buf + 4u, fpst::kFrameMarker.data(), fpst::kFrameMarker.size()) == 0);
        fpst::envelope back;
        std::string error;
        size_t m = 0;
        FPENV_CHECK(fpst::decode_framed_envelope(buf, sizeof(buf), &back, &m, &error));
        FPENV_CHECK(m == sizeof(buf));
        FPENV_CHECK(fpst::match_envelope(back, scope, kPolicy, kNonce0, kNonce1, kSerial, kAdapt, &error));
    }
    {
        uint8_t legacy[fpst::kFramedBytes];
        std::memset(legacy, 0, sizeof(legacy));
        const uint32_t arch_len = 4;
        std::memcpy(legacy, &arch_len, sizeof(arch_len));
        std::memcpy(legacy + 4, "gpt2", 4);
        fpst::envelope out;
        std::string error;
        FPENV_CHECK(!fpst::decode_framed_envelope(legacy, sizeof(legacy), &out, nullptr, &error));
        FPENV_CHECK(!error.empty());
    }
    {
        const fpst::envelope good = fpst::make_envelope(fpst::kScopeFull, kPolicy, kNonce0, kNonce1, kSerial, kAdapt);
        uint8_t raw[fpst::kEnvelopeBytes];
        FPENV_CHECK(fpst::encode_envelope(good, raw, sizeof(raw)));
        uint8_t framed[fpst::kFramedBytes];
        std::memcpy(framed, raw, sizeof(raw));
        std::memset(framed + sizeof(raw), 0, sizeof(framed) - sizeof(raw));
        fpst::envelope out;
        std::string error;
        FPENV_CHECK(!fpst::decode_framed_envelope(framed, sizeof(framed), &out, nullptr, &error));
    }
    {
        const fpst::envelope good = fpst::make_envelope(fpst::kScopeFull, kPolicy, kNonce0, kNonce1, kSerial, kAdapt);
        uint8_t buf[fpst::kFramedBytes];
        FPENV_CHECK(fpst::encode_framed_envelope(good, buf, sizeof(buf)));
        fpst::envelope out;
        std::string error;
        FPENV_CHECK(!fpst::decode_framed_envelope(buf, fpst::kFramedBytes - 1, &out, nullptr, &error));
        FPENV_CHECK(!fpst::decode_framed_envelope(nullptr, 0, &out, nullptr, &error));
        FPENV_CHECK(!fpst::encode_framed_envelope(good, buf, fpst::kFramedBytes - 1));
    }
}

static void test_describe_is_bounded() {
    const fpst::envelope env = fpst::make_envelope(fpst::kScopeFull, kPolicy, kNonce0, kNonce1, kSerial, kAdapt);
    const std::string d = fpst::describe_envelope(env);
    FPENV_CHECK(!d.empty());
    FPENV_CHECK(d.size() < 192u);
}

inline int run_tests() {
    s_failures = 0;
    test_round_trip_full_and_seq();
    test_rejects_legacy_and_corrupt();
    test_match_domains();
    test_cache_key_discrimination();
    test_framed_round_trip_and_rejects();
    test_describe_is_bounded();
    return s_failures;
}

} // namespace flashprefill_state_envelope_tests

#undef FPENV_CHECK
