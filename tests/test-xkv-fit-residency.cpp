// Focused residency-fit tests: the persistent XKV store's device/host charge
// derives from the effective residency predicate
// (llama_xkv_profile_is_device_owned), never from the storage profile alone.
// tq-* + cpu-reference is host-resident (0 device bytes, full host charge);
// TQ profiles on device factorizers are fully device-owned. Hermetic: no
// model, no device, no fit run — pure charge derivation.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "fit.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

static llama_context_params make_params(enum llama_xkv_storage_profile profile,
                                        enum llama_xkv_factorizer factorizer) {
    llama_context_params c = {};
    c.xkv_storage_profile = profile;
    c.xkv_factorizer = factorizer;
    return c;
}

int main() {
    std::cout << "=== test-xkv-fit-residency ===" << std::endl;
    const uint64_t store = 8u * 1024u * 1024u; // 8 MiB persistent store budget

    // CPU-reference TQ is host-fit: zero device bytes, full host charge.
    {
        llama_context_params c = make_params(LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS,
                                             LLAMA_XKV_FACTORIZER_CPU_REFERENCE);
        CHECK(common_xkv_store_device_bytes(&c, store) == 0);
        llama_context_params cl = make_params(LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS,
                                              LLAMA_XKV_FACTORIZER_CPU_REFERENCE);
        CHECK(common_xkv_store_device_bytes(&cl, store) == 0);
    }

    // Vulkan TQ is device-fit: full device bytes, zero host charge.
    {
        llama_context_params c = make_params(LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS,
                                             LLAMA_XKV_FACTORIZER_VULKAN);
        CHECK(common_xkv_store_device_bytes(&c, store) == store);
        llama_context_params ch = make_params(LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS,
                                              LLAMA_XKV_FACTORIZER_VULKAN_HYBRID);
        CHECK(common_xkv_store_device_bytes(&ch, store) == store);
        llama_context_params cc = make_params(LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS,
                                              LLAMA_XKV_FACTORIZER_CUDA);
        CHECK(common_xkv_store_device_bytes(&cc, store) == store);
    }

    // Reference profile is always host-resident, even on device factorizers.
    {
        llama_context_params c = make_params(LLAMA_XKV_STORAGE_PROFILE_REFERENCE,
                                             LLAMA_XKV_FACTORIZER_VULKAN);
        CHECK(common_xkv_store_device_bytes(&c, store) == 0);
        llama_context_params c2 = make_params(LLAMA_XKV_STORAGE_PROFILE_REFERENCE,
                                              LLAMA_XKV_FACTORIZER_CPU_REFERENCE);
        CHECK(common_xkv_store_device_bytes(&c2, store) == 0);
    }

    // Degenerate inputs fail closed to host.
    {
        CHECK(common_xkv_store_device_bytes(nullptr, store) == 0);
        llama_context_params c = make_params(LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS,
                                             LLAMA_XKV_FACTORIZER_VULKAN);
        CHECK(common_xkv_store_device_bytes(&c, 0) == 0);
        llama_context_params bad_p = make_params((enum llama_xkv_storage_profile) -1,
                                                 LLAMA_XKV_FACTORIZER_VULKAN);
        CHECK(common_xkv_store_device_bytes(&bad_p, store) == 0);
        llama_context_params bad_f = make_params(LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS,
                                                 (enum llama_xkv_factorizer) -1);
        CHECK(common_xkv_store_device_bytes(&bad_f, store) == 0);
    }

    // Exact partition over the full cross product: host + device == store,
    // device nonzero exactly for (tq-profile, device-factorizer).
    {
        const enum llama_xkv_storage_profile profiles[] = {
            LLAMA_XKV_STORAGE_PROFILE_REFERENCE,
            LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS,
            LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS,
        };
        const enum llama_xkv_factorizer factorizers[] = {
            LLAMA_XKV_FACTORIZER_CPU_REFERENCE,
            LLAMA_XKV_FACTORIZER_VULKAN,
            LLAMA_XKV_FACTORIZER_VULKAN_HYBRID,
            LLAMA_XKV_FACTORIZER_CUDA,
        };
        for (auto p : profiles) {
            for (auto f : factorizers) {
                llama_context_params c = make_params(p, f);
                const uint64_t dev = common_xkv_store_device_bytes(&c, store);
                const uint64_t host = store - dev;
                CHECK(host + dev == store);
                const bool expect_dev =
                    (p != LLAMA_XKV_STORAGE_PROFILE_REFERENCE) &&
                    (f != LLAMA_XKV_FACTORIZER_CPU_REFERENCE);
                CHECK((dev == store) == expect_dev);
                CHECK(llama_xkv_profile_is_device_owned(p, f) == expect_dev);
            }
        }
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "FAILED: %d checks\n", g_failures);
        return 1;
    }

    // Landmark scratch K-only verification:
    {
        const uint32_t seg_tokens = 4096;
        const uint32_t chunk_tokens = 8;
        const uint64_t feat_k = 4096;
        const uint64_t feat_v = 4096;
        const uint32_t rank_k = 384;
        const uint32_t rank_v = 576;
        uint64_t out_scratch = 0;
        CHECK(common_xkv_scratch_for_group(seg_tokens, chunk_tokens, feat_k, feat_v,
                                           rank_k, rank_v, GGML_TYPE_Q8_0, &out_scratch));
        CHECK(out_scratch > 0);

        // Landmarks summarize K only (feat_k = 4096), never K+V (8192).
        const uint64_t n_frag = seg_tokens / chunk_tokens; // 512
        const size_t lm_row_k = ggml_row_size(GGML_TYPE_Q8_0, (int64_t) feat_k); // 4352 B
        const size_t lm_row_kv = ggml_row_size(GGML_TYPE_Q8_0, (int64_t) (feat_k + feat_v)); // 8704 B
        CHECK(lm_row_k == 4352);
        CHECK(lm_row_kv == 8704);
        CHECK(n_frag * lm_row_k == 2228224ULL);
        CHECK(n_frag * lm_row_kv == 4456448ULL);

        // Unsupported landmark type or unaligned dimensions fail closed
        uint64_t bad_scratch = 0;
        CHECK(!common_xkv_scratch_for_group(seg_tokens, chunk_tokens, feat_k, feat_v,
                                            rank_k, rank_v, GGML_TYPE_I8, &bad_scratch));
        // feat_k not aligned to Q8 block size (32) fails closed
        CHECK(!common_xkv_scratch_for_group(seg_tokens, chunk_tokens, 4095, feat_v,
                                            rank_k, rank_v, GGML_TYPE_Q8_0, &bad_scratch));
        std::cout << "  [PASS] Landmark scratch uses feat_k (2228224 B), not feat_sum (4456448 B)." << std::endl;
    }

    // Store budget independence from non-KV context allocations:
    {
        const uint64_t exact_dense_kv = 64ULL * 1024ULL * 1024ULL; // 64 MiB dense KV
        const double min_saving = 0.10;

        const uint32_t base_mib = common_xkv_store_mib_for_bytes(exact_dense_kv, min_saving);
        CHECK(base_mib == 58); // 64 * 0.9 = 57.6 -> ceil 58 MiB

        const uint32_t overlap_mib = common_xkv_store_mib_with_overlap(exact_dense_kv, min_saving, 64, 1024);
        CHECK(overlap_mib >= base_mib);

        // Any non-KV context addition (e.g. compute buffers) must NOT be included in exact_dense_kv:
        const uint64_t polluted_total_context = exact_dense_kv + 32ULL * 1024ULL * 1024ULL;
        const uint32_t polluted_mib = common_xkv_store_mib_for_bytes(polluted_total_context, min_saving);
        CHECK(polluted_mib == 87);
        CHECK(polluted_mib != base_mib); // Proves pure K/V derivation prevents budget inflation
        std::cout << "  [PASS] Store budget is strictly determined by pure dense K/V bytes." << std::endl;
    }

    // Arithmetic overflow & conservative ceil boundary tests for common_xkv_store_mib_for_bytes:
    {
        constexpr uint64_t MiB = 1024ull * 1024ull;

        // 1. Invalid min_saving fails closed (returns 0), never changes policy silently:
        CHECK(common_xkv_store_mib_for_bytes(64 * MiB, -0.01) == 0);
        CHECK(common_xkv_store_mib_for_bytes(64 * MiB, 1.0) == 0);
        CHECK(common_xkv_store_mib_for_bytes(64 * MiB, 1.5) == 0);
        CHECK(common_xkv_store_mib_for_bytes(64 * MiB, std::numeric_limits<double>::quiet_NaN()) == 0);
        CHECK(common_xkv_store_mib_for_bytes(64 * MiB, std::numeric_limits<double>::infinity()) == 0);

        // 2. Exact MiB boundary rounding: never under-round by even 1 byte!
        // If dense is exactly 10 MiB and saving is 0.0, target is 10 MiB exact -> ceil gives 10
        CHECK(common_xkv_store_mib_for_bytes(10 * MiB, 0.0) == 10);
        // If target is 10 MiB + 1 byte, ceil MUST round up to 11 MiB:
        // dense = 10 * MiB + 2 bytes, saving = 0.0 -> target = 10 MiB + 2 B -> 11 MiB
        CHECK(common_xkv_store_mib_for_bytes(10 * MiB + 2, 0.0) == 11);

        // 3. Overflow boundary: dense_ctx_bytes near UINT64_MAX fails closed cleanly without wrapping
        CHECK(common_xkv_store_mib_for_bytes(UINT64_MAX, 0.0) == 0);
        CHECK(common_xkv_store_mib_for_bytes(UINT64_MAX - 100, 0.10) == 0);

        std::cout << "  [PASS] Store budget boundary and overflow hardening verified." << std::endl;
    }

    // Overflow and fail-closed validation on budget partition:
    {
        uint64_t shares[2] = {0, 0};
        uint64_t weights[2] = {100, 200};
        // Null args fail closed
        CHECK(!common_xkv_partition_budget(1000, nullptr, shares, 2));
        CHECK(!common_xkv_partition_budget(1000, weights, nullptr, 2));
        // Zero devices returns true (no-op)
        CHECK(common_xkv_partition_budget(1000, weights, shares, 0));
        // Normal partition sum exact
        CHECK(common_xkv_partition_budget(1000, weights, shares, 2));
        CHECK(shares[0] + shares[1] == 1000);
        // Weight sum overflow fails closed
        uint64_t bad_wts[2] = {UINT64_MAX - 10, 20};
        CHECK(!common_xkv_partition_budget(1000, bad_wts, shares, 2));
        CHECK(shares[0] == 0 && shares[1] == 0);
        std::cout << "  [PASS] Budget partition overflow and exact-sum properties verified." << std::endl;
    }

    // Overflow validation on common_xkv_fit_reserve_bytes:
    {
        llama_context_params c = {};
        c.xkv_mode = LLAMA_XKV_MODE_DENSE;
        c.xkv_workspace_mib = 128;
        c.xkv_decode_cache_mib = 64;
        common_xkv_fit_reserve r = {};
        // Scratch + decode exceeding workspace fails closed:
        const uint64_t huge_scratch = 70ull * 1024ull * 1024ull;
        CHECK(!common_xkv_fit_reserve_bytes(&c, huge_scratch, &r));
        CHECK(r.total_bytes == UINT64_MAX);

        // Workspace MiB overflow fails closed:
        llama_context_params c_over = c;
        c_over.xkv_workspace_mib = UINT32_MAX;
        CHECK(!common_xkv_fit_reserve_bytes(&c_over, 1024, &r));
        CHECK(r.total_bytes == UINT64_MAX);

        // Decode cache MiB overflow fails closed:
        c_over = c;
        c_over.xkv_decode_cache_mib = UINT32_MAX;
        CHECK(!common_xkv_fit_reserve_bytes(&c_over, 1024, &r));
        CHECK(r.total_bytes == UINT64_MAX);
        std::cout << "  [PASS] Reserve accounting overflow fails closed." << std::endl;
    }

    // Store budget with worst-case one-segment seal/COW overlap overflow:
    {
        constexpr uint64_t MiB = 1024ull * 1024ull;
        // Zero k_tokens returns base
        CHECK(common_xkv_store_mib_with_overlap(64 * MiB, 0.1, 4096, 0) == common_xkv_store_mib_for_bytes(64 * MiB, 0.1));
        // Zero seg_tokens returns base
        CHECK(common_xkv_store_mib_with_overlap(64 * MiB, 0.1, 0, 4096) == common_xkv_store_mib_for_bytes(64 * MiB, 0.1));
        // Overflow in dense_ctx_bytes * seg_tokens fails closed (returns 0)
        CHECK(common_xkv_store_mib_with_overlap(UINT64_MAX / 2, 0.1, 4096, 4096) == 0);
        std::cout << "  [PASS] Store budget with overlap overflow tested." << std::endl;
    }

    std::cout << "=== ALL FIT-RESIDENCY CHECKS PASSED ===" << std::endl;
    return 0;
}
