#include "llama.h"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// One round is 1, 3, 2, 4 real tokens. No dummy rows are submitted.
// Zero-output arm: a step that marks no token as output binds the
// output-gather closure to width 0 in place. The arm requires the step to
// complete, refused_nout and lane_declines to stay put, exactly one more
// in-place patch, bound_n_outputs to be 0, build_n_outputs and the graph
// build count to be unchanged, and the next two-output step to return finite
// logits. The falsifiers are the three that move together when the zero step
// rebuilds: patch_binds, bound_n_outputs and the build count.
int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s model.gguf [rounds=50]\n", argv[0]);
        return 2;
    }
    char * end = nullptr;
    errno = 0;
    long rounds = argc == 3 ? std::strtol(argv[2], &end, 10) : 50;
    if (errno || (argc == 3 && (end == argv[2] || *end)) || rounds < 1 || rounds > INT_MAX / 10) {
        std::fprintf(stderr, "FAIL: invalid round count\n");
        return 2;
    }
    const int total_tokens = (int) rounds * 10;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    ggml_backend_dev_t devices[] = { nullptr };
    mp.devices = devices;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        std::fprintf(stderr, "FAIL: model load\n");
        llama_backend_free();
        return 1;
    }

    auto cp = llama_context_default_params();
    cp.n_ctx = cp.n_ctx_kv = total_tokens;
    cp.n_batch = cp.n_ubatch = cp.n_outputs_max = 4;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 1;
    cp.kv_unified = true;
    cp.offload_kqv = cp.op_offload = false;
    cp.no_perf = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.embeddings = false;
    cp.cb_eval = nullptr;
    cp.cb_eval_user_data = nullptr;

    setenv("GGML_TP5_UNION_BINDER", "1", 1);
    setenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK", "1", 1);
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    llama_context * on = llama_init_from_model(model, cp);
    setenv("GGML_TP5_UNION_BINDER", "0", 1);
    llama_context * off = llama_init_from_model(model, cp);
    bool ok = on && off;
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int dim = llama_model_n_embd_out(model);
    if (!ok || vocab <= 3 || dim <= 0) {
        std::fprintf(stderr, "FAIL: contexts or nonpositive output dimensions\n");
        ok = false;
    }
    const uint64_t init_on = on ? llama_graph_build_count(on) : 0;
    const uint64_t init_off = off ? llama_graph_build_count(off) : 0;
    std::fprintf(stderr, "init_builds on=%llu off=%llu requested_tokens=%d rounds=%ld\n",
                 (unsigned long long) init_on, (unsigned long long) init_off, total_tokens, rounds);
    llama_union_diagnostics d = {}, off_d = {};
    const bool off_has_state = off && llama_get_union_diagnostics(off, &off_d);
    if (ok && (!llama_get_union_diagnostics(on, &d) || !d.enabled || d.envelope_disabled ||
               (off_has_state && off_d.enabled))) {
        std::fprintf(stderr, "FAIL: ON/OFF binder state\n");
        ok = false;
    }
    const uint64_t fallback_base = d.rebuild_fallbacks;
    // padded-lane fallback baseline: the exact path is "selected and delivered"
    // only while this does not move and bound_n_kv is the true used span
    const uint64_t pad_base = d.pad_fallbacks;
    uint64_t previous_binds = d.patch_binds;
    const int widths[] = { 1, 3, 2, 4 };
    std::vector<float> a(dim > 0 ? dim : 0), b(dim > 0 ? dim : 0);
    int pos = 0;
    int checked_rows = 0;
    for (int step = 0; ok && step < rounds * 4; ++step) {
        const int width = widths[step % 4];
        llama_batch batch = llama_batch_init(width, 0, 1);
        if (!llama_batch_is_complete(&batch, width, false, true)) {
            llama_batch_free(batch);
            batch = {};
            ok = false;
            break;
        }
        for (int i = 0; i < width; ++i) {
            batch.token[i] = 3 + (int) (((int64_t) (pos + i) * 7 + 1) % (vocab - 3));
            batch.pos[i] = pos + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        batch.n_tokens = width;
        const int rc_on = llama_decode(on, batch);
        const int rc_off = llama_decode(off, batch);
        llama_batch_free(batch);
        const uint64_t on_now = llama_graph_build_count(on);
        const uint64_t off_now = llama_graph_build_count(off);
        bool have_stats = llama_get_union_diagnostics(on, &d);
        const bool off_has_state_now = llama_get_union_diagnostics(off, &off_d);
        std::fprintf(stderr, "step=%d width=%d on_total=%llu on_decode=%llu off_total=%llu off_decode=%llu binds=%llu lane_declines=%llu rebuild_fallbacks=%llu bound_tokens=%u bound_n_kv=%u pad_fallbacks=%llu\n",
                     step, width, (unsigned long long) on_now, (unsigned long long) (on_now - init_on),
                     (unsigned long long) off_now, (unsigned long long) (off_now - init_off),
                     (unsigned long long) d.patch_binds, (unsigned long long) d.lane_declines,
                     (unsigned long long) d.rebuild_fallbacks, d.bound_tokens,
                     d.bound_n_kv, (unsigned long long) d.pad_fallbacks);
        // bound_n_kv must be the true used span after this step, which is the
        // cumulative token count pos + width: the single sequence keeps every
        // submitted token resident, and the stock padded width would not equal
        // it (the schedule never reaches 256).
        //
        // pad_fallbacks must not move: the exact lane is armed on this arm, and
        // a layout refusal there is a hard failure rather than a padded
        // fallback, so any movement means the padded lane came back.
        // The record build is at the live widths and applies no patch when
        // nothing has moved yet (union_bind_is_noop), so step 0 is not
        // required to bind. Every later step moves the token and KV axes and
        // must bind.
        if (rc_on || rc_off || on_now < init_on || on_now - init_on != 1 ||
            !have_stats || !d.enabled || d.envelope_disabled || !d.record_valid || !d.token_map_valid ||
            (off_has_state_now && off_d.enabled) || d.bound_tokens != (uint32_t) width ||
            d.bound_n_kv != (uint32_t) (pos + width) || d.pad_fallbacks != pad_base ||
            (step > 0 && d.patch_binds <= previous_binds) ||
            d.rebuild_fallbacks != fallback_base) {
            std::fprintf(stderr, "FAIL: decode/build/bind/fallback/exact step=%d on_rc=%d off_rc=%d bound_n_kv=%u want=%d pad_fallbacks=%llu want=%llu\n",
                         step, rc_on, rc_off, d.bound_n_kv, pos + width,
                         (unsigned long long) d.pad_fallbacks, (unsigned long long) pad_base);
            ok = false;
            break;
        }
        previous_binds = d.patch_binds;
        for (int i = 0; i < width; ++i) {
            const float * logits_on = llama_get_logits_ith(on, i);
            const float * logits_off = llama_get_logits_ith(off, i);
            const bool activation_on = llama_copy_last_llama_activation(on, i, a.data(), a.size());
            const bool activation_off = llama_copy_last_llama_activation(off, i, b.data(), b.size());
            if (!logits_on || !logits_off || !activation_on || !activation_off ||
                std::memcmp(logits_on, logits_off, (size_t) vocab * sizeof(float)) ||
                std::memcmp(a.data(), b.data(), (size_t) dim * sizeof(float))) {
                std::fprintf(stderr, "FAIL: logits or final normalized activation step=%d row=%d activation_on=%d activation_off=%d\n",
                             step, i, activation_on, activation_off);
                ok = false;
                break;
            }
            ++checked_rows;
        }
        pos += width;
    }
    if (ok && (pos != total_tokens || checked_rows != total_tokens)) ok = false;
    // A zero-output step is an ordinary shrink: it patches the closure to
    // zero rows and does not rebuild. A fresh binder context keeps this arm
    // out of the n_tokens trajectory above.
    if (ok) {
        setenv("GGML_TP5_UNION_BINDER", "1", 1);
        llama_context * zc = llama_init_from_model(model, cp);
        if (zc == nullptr) {
            std::fprintf(stderr, "FAIL: nout-zero context\n");
            ok = false;
        } else {
            auto decode_two = [&](llama_pos p, int logits_flag) {
                llama_batch b = llama_batch_init(2, 0, 1);
                if (!llama_batch_is_complete(&b, 2, false, true)) {
                    llama_batch_free(b);
                    return -1;
                }
                for (int i = 0; i < 2; ++i) {
                    b.token[i] = 3 + (int) (((int64_t) (p + i) * 7 + 1) % (vocab - 3));
                    b.pos[i] = p + i;
                    b.n_seq_id[i] = 1;
                    b.seq_id[i][0] = 0;
                    b.logits[i] = logits_flag;
                }
                b.n_tokens = 2;
                const int rc = llama_decode(zc, b);
                llama_batch_free(b);
                return rc;
            };
            const int rc_seed = decode_two(0, 1);
            llama_nkv_runtime_stats before = {};
            llama_union_diagnostics before_d = {};
            const bool have_before = llama_get_nkv_runtime_stats(zc, &before) &&
                                     llama_get_union_diagnostics(zc, &before_d);
            const uint64_t builds_before_zero = llama_graph_build_count(zc);
            const int rc_zero = decode_two(2, 0);
            llama_nkv_runtime_stats after = {};
            llama_union_diagnostics after_d = {};
            const bool have_after = llama_get_nkv_runtime_stats(zc, &after) &&
                                    llama_get_union_diagnostics(zc, &after_d);
            const bool arm_ok = rc_seed == 0 && have_before && rc_zero == 0 && have_after &&
                after.runtime_n_outputs == 0 &&
                // the zero step BINDS: no refusal, one more in-place patch, and
                // the bound width is 0 while the build width did not move
                after.refused_nout == before.refused_nout &&
                after_d.lane_declines == before_d.lane_declines &&
                after_d.patch_binds == before_d.patch_binds + 1 &&
                after.bound_n_outputs == 0 &&
                after.build_n_outputs == before.build_n_outputs &&
                llama_graph_build_count(zc) == builds_before_zero;
            if (!arm_ok) {
                std::fprintf(stderr, "FAIL: NOUT_ZERO_BIND seed=%d zero=%d refused=%llu/%llu lane_declines=%llu/%llu patch_binds=%llu/%llu bound=%u build=%u/%u builds=%llu/%llu runtime=%u\n",
                        rc_seed, rc_zero,
                        (unsigned long long) before.refused_nout,
                        (unsigned long long) after.refused_nout,
                        (unsigned long long) before_d.lane_declines,
                        (unsigned long long) after_d.lane_declines,
                        (unsigned long long) before_d.patch_binds,
                        (unsigned long long) after_d.patch_binds,
                        after.bound_n_outputs, after.build_n_outputs, before.build_n_outputs,
                        (unsigned long long) builds_before_zero,
                        (unsigned long long) llama_graph_build_count(zc),
                        after.runtime_n_outputs);
                ok = false;
            } else {
                const int rc_after = decode_two(4, 1);
                const float * row = llama_get_logits_ith(zc, 0);
                bool finite = row != nullptr;
                for (int i = 0; finite && i < vocab; ++i) finite = std::isfinite(row[i]);
                if (rc_after != 0 || !finite) {
                    std::fprintf(stderr, "FAIL: NOUT_ZERO_BIND follow-up rc=%d finite=%d\n", rc_after, (int) finite);
                    ok = false;
                } else {
                    std::fprintf(stderr, "NOUT_ZERO_BIND ok=1 bound_n_outputs=0 build_n_outputs=%u refused_nout=%llu patch_binds=%llu builds=%llu\n",
                            after.build_n_outputs,
                            (unsigned long long) after.refused_nout,
                            (unsigned long long) after_d.patch_binds,
                            (unsigned long long) llama_graph_build_count(zc));
                }
            }
            llama_free(zc);
        }
    }
    std::fprintf(stderr, "%s: checked_rows=%d expected=%d\n", ok ? "PASS" : "FAIL", checked_rows, total_tokens);
    if (on) llama_free(on);
    if (off) llama_free(off);
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
