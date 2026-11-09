// Real llama_state_seq_get_data_ext/set_data_ext round trip on a CPU context.
// Host flags restore the exact host state bytes; ON_DEVICE flags capture only
// metadata while the tensor snapshot lives in ctx->mem_storage[seq]. A device
// restore after a mutation must restore the exact host state. A device restore
// without a matching snapshot must return 0, not abort.
// Set TP5_TEST_MODEL to a small GGUF; without it the test reports SKIP.

#include <llama.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int fails = 0;
static void check(bool ok, const char * name) { std::printf("%s %s\n", ok ? "PASS" : "FAIL", name); if (!ok) ++fails; }

static int decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch b = llama_batch_init(1, 0, 1);
    if (!b.token || !b.pos || !b.n_seq_id || !b.seq_id || !b.logits) { llama_batch_free(b); return -1; }
    b.n_tokens = 1; b.token[0] = tok; b.pos[0] = pos; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc;
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : std::getenv("TP5_TEST_MODEL");
    if (path == nullptr || path[0] == '\0') { std::printf("SKIP no model\n"); return 0; }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (model == nullptr) { std::printf("FAIL load model\n"); return 1; }
    auto cp = llama_context_default_params();
    cp.n_ctx = 64; cp.n_batch = 8; cp.n_ubatch = 8; cp.n_seq_max = 2; cp.n_threads = 1; cp.n_threads_batch = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) { std::printf("FAIL context\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::printf("n_vocab=%d\n", n_vocab);
    llama_token tok = llama_vocab_bos(vocab);
    if (tok < 0 || tok >= n_vocab) tok = n_vocab > 1 ? 1 : 0;
    int rc = decode_one(ctx, tok, 0);
    std::printf("decode0 rc=%d\n", rc);
    if (rc != 0) { std::printf("FAIL decode0\n"); return 1; }

    const size_t hsz = llama_state_seq_get_size_ext(ctx, 0, 0);
    std::printf("host size=%zu\n", hsz);
    std::vector<uint8_t> h1(hsz), h2(hsz);
    const size_t hn = llama_state_seq_get_data_ext(ctx, h1.data(), h1.size(), 0, 0);
    std::printf("host get=%zu\n", hn);
    check(hn == hsz && hn > 0, "host get returns the requested size");

    rc = decode_one(ctx, tok, 1);
    std::printf("decode1 rc=%d\n", rc);
    const size_t hset = llama_state_seq_set_data_ext(ctx, h1.data(), hn, 0, 0);
    std::printf("host set=%zu\n", hset);
    check(hset == hn, "host set returns the captured size");
    const size_t hn2 = llama_state_seq_get_data_ext(ctx, h2.data(), h2.size(), 0, 0);
    std::printf("host get2=%zu equal=%d\n", hn2, hn2 == hn && std::memcmp(h1.data(), h2.data(), hn) == 0);
    check(hn2 == hn && std::memcmp(h1.data(), h2.data(), hn) == 0, "host round trip restores the exact bytes");

    const auto dev = LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
    std::vector<uint8_t> hb(hsz);
    const size_t hbn = llama_state_seq_get_data_ext(ctx, hb.data(), hb.size(), 0, 0);
    check(hbn == hsz, "host capture B");
    const size_t dsz = llama_state_seq_get_size_ext(ctx, 0, dev);
    std::printf("dev size=%zu\n", dsz);
    std::vector<uint8_t> d1(dsz);
    const size_t dn = llama_state_seq_get_data_ext(ctx, d1.data(), d1.size(), 0, dev);
    std::printf("dev get=%zu\n", dn);
    check(dn == dsz && dn > 0, "device get returns the requested metadata size");

    rc = decode_one(ctx, tok, 2);
    std::printf("decode2 rc=%d\n", rc);
    const size_t dset = llama_state_seq_set_data_ext(ctx, d1.data(), dn, 0, dev);
    std::printf("dev set=%zu\n", dset);
    check(dset == dn, "device set returns the captured metadata size");
    std::vector<uint8_t> h2b(hsz);
    const size_t h2bn = llama_state_seq_get_data_ext(ctx, h2b.data(), h2b.size(), 0, 0);
    std::printf("host after device restore=%zu equal_B=%d\n", h2bn,
                h2bn == hbn && std::memcmp(hb.data(), h2b.data(), hbn) == 0);
    check(h2bn == hbn && std::memcmp(hb.data(), h2b.data(), hbn) == 0,
          "device restore restores the exact host state B");

    // A second device capture for the same seq replaces mem_storage[seq].
    // Restoring the older metadata must be detected and return 0 instead of
    // silently copying the newer tensors.
    rc = decode_one(ctx, tok, 3);
    std::printf("decode3 rc=%d\n", rc);
    const size_t dsz3 = llama_state_seq_get_size_ext(ctx, 0, dev);
    std::vector<uint8_t> d3(dsz3);
    const size_t dn3 = llama_state_seq_get_data_ext(ctx, d3.data(), d3.size(), 0, dev);
    std::printf("dev second get=%zu (size=%zu)\n", dn3, dsz3);
    check(dn3 == dsz3 && dn3 > 0, "device second capture");
    const size_t stale = llama_state_seq_set_data_ext(ctx, d1.data(), dn, 0, dev);
    std::printf("dev stale restore=%zu (expect 0)\n", stale);
    check(stale == 0, "device restore of overwritten metadata returns 0");

    std::vector<uint8_t> missing = d1;
    if (missing.size() >= 8) {
        const int32_t sid = 1;
        std::memcpy(missing.data() + 4, &sid, sizeof(sid));
    }
    const size_t mret = llama_state_seq_set_data_ext(ctx, missing.data(), missing.size(), 0, dev);
    std::printf("dev missing set=%zu (expect 0)\n", mret);
    check(mret == 0, "device restore without a snapshot returns 0");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    std::printf("%s failures=%d\n", fails ? "FAILED" : "OK", fails);
    return fails ? 1 : 0;
}
