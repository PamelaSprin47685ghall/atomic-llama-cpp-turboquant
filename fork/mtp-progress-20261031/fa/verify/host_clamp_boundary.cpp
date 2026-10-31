// verify/host_clamp_boundary.cpp — CPU boundary check for the production width
// clamp in build_attn_inp_kq_mask() (src/llama-graph.cpp), the single place
// where the KQ-mask KV width is decided for the plain-FA path.
//
// NOT EXECUTED in this session (user discipline: code + static review only).
// Exact build/run commands: ../RUNBOOK.md.
//
// The static production function is reached by including the unchanged
// translation unit. The production class has no public way to set the cache
// width of a bare context, so the harness uses a test-only access shim
// (#define private public) around llama-kv-cache.h. This does not change any
// production file, and layout/behavior of the class is untouched; it is the
// same class from the same header.
//
// Covered: widths 1/255/256/257/511/512, illegal lengths (-2, -1, 0, 513,
// INT32_MAX, INT32_MIN), multi-column layouts (2 streams, and unified 1 stream
// with 4 tokens), mask dtype selection (F16 for flash attention, F32 otherwise),
// tensor name and 4D shape.

#include <bits/stdc++.h>

#include "llama.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-hparams.h"

#define private public
#include "llama-kv-cache.h"
#undef private

#include "llama-graph.cpp"

namespace {

int g_fail = 0;

void expect(bool ok, const std::string & what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) {
        ++g_fail;
    }
}

llama_kv_cache_context make_ctx(int32_t cache_width) {
    llama_kv_cache_context ctx(LLAMA_MEMORY_STATUS_SUCCESS);
    ctx.n_kv = cache_width;
    return ctx;
}

struct probe {
    int32_t read;   // n_kv_read argument passed to the production builder
    int32_t want;   // expected mask ne[0]
};

void run_widths(int32_t cache_width, const std::vector<probe> & probes, bool unified,
                bool flash_attn) {
    ggml_init_params ip = { 16u * 1024u * 1024u, nullptr, true };
    ggml_context * gctx = ggml_init(ip);
    GGML_ASSERT(gctx != nullptr);

    llama_ubatch ubatch{};
    ubatch.n_tokens   = 4;
    ubatch.n_pos      = 1;
    ubatch.n_seqs     = 1;
    ubatch.n_seqs_unq = 2;

    llama_cparams cparams{};
    cparams.flash_attn = flash_attn;
    cparams.kv_unified = unified;

    const int64_t n_stream = unified ? 1 : 2;
    const int64_t n_rows   = unified ? 4 : 2;

    llama_kv_cache_context ctx = make_ctx(cache_width);

    for (const auto & p : probes) {
        ggml_tensor * mask = build_attn_inp_kq_mask(gctx, &ctx, ubatch, cparams, p.read);

        const std::string tag = "cache_width=" + std::to_string(cache_width) +
                                " read=" + std::to_string(p.read) +
                                (unified ? " unified" : " multi-stream") +
                                (flash_attn ? " fa" : " no-fa");

        bool ok = mask != nullptr;
        ok = ok && mask->ne[0] == p.want;
        ok = ok && mask->ne[1] == n_rows;
        ok = ok && mask->ne[2] == 1;
        ok = ok && mask->ne[3] == n_stream;
        ok = ok && mask->type == (flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32);
        ok = ok && std::string(ggml_get_name(mask)) == "attn_inp_kq_mask";
        expect(ok, tag + " -> ne0=" + std::to_string(mask ? mask->ne[0] : -1) +
                   " want=" + std::to_string(p.want));
    }

    ggml_free(gctx);
}

} // namespace

int main() {
    const int32_t W = 512;

    // exact in-range widths, including the 255/256/257 and 511/512 boundaries
    std::vector<probe> in_range;
    for (int32_t v : { 1, 255, 256, 257, 511, 512 }) {
        in_range.push_back({ v, v });
    }

    // illegal / out-of-range lengths: everything outside (0, cache_width] falls
    // back to the stock cache width (no error, no narrowing)
    std::vector<probe> illegal = {
        { -2,          W },
        { -1,          W },
        {  0,          W },
        { W + 1,       W },
        { INT32_MAX,   W },
        { INT32_MIN,   W },
    };

    for (bool unified : { false, true }) {
        for (bool fa : { true, false }) {
            run_widths(W, in_range, unified, fa);
            run_widths(W, illegal,  unified, fa);
        }
    }

    // small cache: the clamp boundary is read == cache width == 1
    run_widths(1, { { 1, 1 }, { 0, 1 }, { 2, 1 }, { -1, 1 } }, false, true);

    // the same widths, FA-only (the lane the gate arms)
    run_widths(256, { { 255, 255 }, { 256, 256 }, { 257, 256 }, { 0, 256 } }, false, true);
    run_widths(257, { { 256, 256 }, { 257, 257 }, { 258, 257 } }, false, true);

    std::printf("%s host_clamp_boundary failures=%d\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
