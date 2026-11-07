// Does the binder's in-place narrowing change the arithmetic?
//
// The binder patches a live graph's attention tensors in place: it rewrites
// ne[], nb[] and view_offs so a graph built at the envelope width executes at
// the true used span (llama-context.cpp, union_rebind_apply). This test is the
// regression guard for that patch. It is NOT the equality argument: a complete
// patch is an identity by construction, so the STOCK arm below can only check
// that the patch is complete. The substantive claim - that the narrowed graph
// runs the same kernels with the same accumulation tree as a graph built at
// that width - rests on the selection-site table in ANALYSIS-EQUALITY.md §C and
// on the ordered kernels' sequential accumulation, not on this file.
//
// Arms, none of which marks anything (REDUCE_PAD_TAIL / MASKED_TAIL_GEMM are
// gone from the n_kv lane):
//
//   STOCK   built at REF_W, patched COMPLETELY to width n (ne[] and nb[]),
//           versus built at n. Requirement: agree on every case. Because the
//           patch is complete this is an identity; it catches a binder that
//           forgets to patch at all.
//   STRIDE  built at REF_W, patched INCOMPLETELY: ne[] narrowed, nb[] left at
//           the envelope stride. That is the plausible binder bug - a missed
//           stride update - and it MUST differ somewhere, or the comparison
//           cannot see a stride defect. ggml asserts on the inconsistent
//           layout (assert(ggml_is_contiguous(dst)) in the softmax), so this
//           arm runs in a child process and an abnormal exit counts as
//           detection (differ=1 reason=abort).
//   OLD     two unmarked widths straddling a tinyBLAS tile boundary
//           (m % 16 / 8 / 4, m = ne01). Must differ somewhere, or the
//           comparison cannot see a selection difference.
//
// No model, no GPU, no environment variable.
//
// Output:
//   KERNEL_SEL_SITE   name=<site> arm=<arm> tensor=<t> dim=<d> value=<v> pred=<0|1>
//   KERNEL_SEL_STOCK  name=<arm> n=<n> differ=<0|1>
//   KERNEL_SEL_STRIDE name=<arm> n=<n> differ=<0|1> reason=<detected|none>
//   KERNEL_SEL_OLD    name=<arm> n=<n> n2=<n2> differ=<0|1>
//   KERNEL_SEL_END    ok=<0|1> stock=<a>/<r> stride=<d>/<r> old=<d>/<r>
// plus one KERNEL_SEL_FAIL line per failing case.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>   // atoi, for the --stride-child argument vector
#include <cstring>
#include <vector>

#if defined(__unix__)
#include <sys/types.h>  // pid_t, fork(): glibc pulls this in via unistd.h, but that is a libc detail
#include <sys/wait.h>
#include <unistd.h>
#endif

// The envelope the patched arm is built at.
static constexpr int REF_W = 256;

// Widths the patch narrows to. Crosses every SIMD step the CPU dots use
// (16/32/64) and the 256-cell bucket.
static const int NARROW_W[] = { 1, 7, 8, 9, 15, 16, 17, 31, 32, 33,
                                127, 128, 129, 249, 250, 255 };
static constexpr int N_NARROW = (int) (sizeof(NARROW_W) / sizeof(NARROW_W[0]));

// `rows` is the value product's src0 row count, and tinyBLAS picks its tile
// from m % 16 / 8 / 4, so both sides of each class are covered.
static const int ROWS[] = { 8, 12, 16, 32 };
static constexpr int N_ROWS = (int) (sizeof(ROWS) / sizeof(ROWS[0]));

// 1 is decode, 4 a small prefill.
static const int QUERIES[] = { 1, 4 };
static constexpr int N_QUERIES = (int) (sizeof(QUERIES) / sizeof(QUERIES[0]));

static float f32_at(int row, int col, int salt) {
    const int v = ((row * 37 + col * 61 + salt * 13) % 251) - 125;
    return (float) v / 128.0f;
}

static ggml_fp16_t f16_at(int row, int col, int salt) {
    return ggml_fp32_to_fp16(f32_at(row, col, salt));
}

enum patch_kind {
    PATCH_NONE = 0,   // built at `n`, nothing to patch
    PATCH_FULL,       // ne[] and nb[] narrowed: the binder's correct patch
    PATCH_NE_ONLY,    // ne[] narrowed, nb[] left at the envelope stride
};

struct chain_out {
    std::vector<float> p;   // [n, queries], query q at q*n
    std::vector<float> out; // [rows, queries], query q at q*rows
    bool ok = false;
};

// Build the non-FA attention chain at `built_at`, optionally patch it to `n`,
// then run it.
static bool compute_chain(ggml_type vtype, ggml_prec prec, int built_at, int n,
                          patch_kind patch, int queries, int rows, chain_out & res) {
    if (n > built_at) return false;

    ggml_init_params ip = { 16u*1024u*1024u, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    ggml_tensor * s    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, built_at, queries);
    ggml_tensor * v    = ggml_new_tensor_2d(ctx, vtype,          built_at, rows);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  built_at, queries);

    // The operand data is laid out for the width the graph is SUPPOSED to
    // execute at, `fill_w`, and not for the allocation width. The metadata
    // (ne[]/nb[]) is what the binder controls, so this is the only layout that
    // makes the three arms mean what they claim:
    //
    //   STOCK   metadata patched to (ne,nb) = (n,n) reads this layout exactly
    //           -> an identity, as claimed.
    //   STRIDE  metadata patched to (n,built_at) reads row r at the envelope
    //           stride while the data sits at the narrow one -> a misread,
    //           which is what a missed stride update does.
    //   skipped metadata left at (built_at,built_at) reads past the live span
    //           into the zeroed tail -> a different softmax, so "the binder
    //           forgot to patch at all" is visible too.
    //
    // The tail is zeroed rather than filled with neutral -inf mask entries and
    // zero values: a neutral tail would make a too-wide read agree with the
    // narrow one and the comparison would be blind to exactly the defect it
    // exists to see. The allocation is memset first so the tail is defined.
    const int fill_w = patch == PATCH_NONE ? built_at : n;
    std::memset(s->data,    0, ggml_nbytes(s));
    std::memset(mask->data, 0, ggml_nbytes(mask));
    std::memset(v->data,    0, ggml_nbytes(v));
    for (int q = 0; q < queries; ++q) {
        for (int i = 0; i < fill_w; ++i) {
            ((float *) ((char *) s->data + q*(size_t) fill_w*sizeof(float) + i*sizeof(float)))[0] =
                f32_at(q, i, 1);
            ((float *) ((char *) mask->data + q*(size_t) fill_w*sizeof(float) + i*sizeof(float)))[0] =
                f32_at(q, i, 2);
        }
    }
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < fill_w; ++i) {
            char * dst = (char *) v->data + r*ggml_row_size(vtype, fill_w) + i*ggml_type_size(vtype);
            if (vtype == GGML_TYPE_F16) {
                *(ggml_fp16_t *) dst = f16_at(r, i, 3);
            } else {
                *(float *) dst = f32_at(r, i, 3);
            }
        }
    }

    ggml_tensor * p   = ggml_soft_max_ext(ctx, s, mask, 1.0f, 0.0f);
    ggml_tensor * out = ggml_mul_mat(ctx, v, p);
    ggml_mul_mat_set_prec(out, prec);

    if (patch != PATCH_NONE) {
        const int64_t w = n;
        const bool strides = (patch == PATCH_FULL);
        // s and mask: [w, queries] F32
        for (int ti = 0; ti < 2; ++ti) {
            ggml_tensor * t = ti == 0 ? s : mask;
            t->ne[0] = w;
            if (strides) {
                t->nb[1] = (size_t) w * sizeof(float);
                t->nb[2] = t->nb[1] * queries;
                t->nb[3] = t->nb[2];
            }
        }
        // v: [w, rows] of vtype
        v->ne[0] = w;
        if (strides) {
            v->nb[1] = ggml_row_size(vtype, w);
            v->nb[2] = v->nb[1] * rows;
            v->nb[3] = v->nb[2];
        }
        // p = softmax(s, mask): [w, queries]
        p->ne[0] = w;
        if (strides) {
            p->nb[1] = (size_t) w * sizeof(float);
            p->nb[2] = p->nb[1] * queries;
            p->nb[3] = p->nb[2];
        }
        // out = mul_mat(v, p): reduction w, output [rows, queries]
        out->ne[0] = rows;
        if (strides) {
            out->nb[1] = (size_t) rows * sizeof(float);
            out->nb[2] = out->nb[1] * queries;
            out->nb[3] = out->nb[2];
        }
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    const bool ran = ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS;
    res.ok = ran;
    if (ran) {
        const int width = patch == PATCH_NONE ? built_at : n;
        res.p.resize((size_t) width * queries);
        res.out.resize((size_t) rows * queries);
        for (int q = 0; q < queries; ++q) {
            std::memcpy(res.p.data()   + (size_t) q * width,
                        (char *) p->data   + q*p->nb[1], (size_t) width * sizeof(float));
            std::memcpy(res.out.data() + (size_t) q * rows,
                        (char *) out->data + q*out->nb[1], (size_t) rows * sizeof(float));
        }
    }
    ggml_free(ctx);
    return ran;
}

// Print the selection predicates with the tensor and dimension each one came
// from, so the trace is derived rather than asserted.
static void emit_sites(const char * arm, int n, int rows) {
    // The value product's MUL_MAT: src0 is v, so ne01 = rows and ne00 = n.
    //
    // tinyBLAS::matmul (sgemm.cpp:495-533) picks its tile from
    //   m % 16 == 0 && m/16 >= nth   -> mnpack<4,6,4>
    //   m % 8  == 0                  -> mnpack<4,6,2>
    //   m % 4  == 0                  -> mnpack<4,6,1>
    // with m = ne01, and rejects the call outright when k % KN != 0 with
    // k = ne00. KN is 4, 8 or 16 by ISA, so "k % 16" is the strictest of the
    // three possible tests, not THE test. Each predicate is named for what it
    // computes.
    std::printf("KERNEL_SEL_SITE name=m_ne01_mod16 arm=%s tensor=v dim=1 value=%d pred=%d\n",
                arm, rows, (rows % 16 == 0) ? 1 : 0);
    std::printf("KERNEL_SEL_SITE name=m_ne01_mod8 arm=%s tensor=v dim=1 value=%d pred=%d\n",
                arm, rows, (rows % 8 == 0) ? 1 : 0);
    std::printf("KERNEL_SEL_SITE name=m_ne01_mod4 arm=%s tensor=v dim=1 value=%d pred=%d\n",
                arm, rows, (rows % 4 == 0) ? 1 : 0);
    std::printf("KERNEL_SEL_SITE name=k_ne00_mod16 arm=%s tensor=v dim=0 value=%d pred=%d\n",
                arm, n, (n % 16 == 0) ? 1 : 0);
    std::printf("KERNEL_SEL_SITE name=k_ne00_mod8 arm=%s tensor=v dim=0 value=%d pred=%d\n",
                arm, n, (n % 8 == 0) ? 1 : 0);
    std::printf("KERNEL_SEL_SITE name=k_ne00_mod4 arm=%s tensor=v dim=0 value=%d pred=%d\n",
                arm, n, (n % 4 == 0) ? 1 : 0);
    // The value product's reduction width is p's ne[0], the same value as
    // v's ne[0] after the patch; printed separately so a patch that moves one
    // and not the other is visible in the trace.
    std::printf("KERNEL_SEL_SITE name=reduce_width arm=%s tensor=p dim=0 value=%d pred=%d\n",
                arm, n, (n % 4 == 0) ? 1 : 0);
}

struct case_result {
    bool ran  = false;
    bool same = false;
};

// STOCK: complete patch versus built-at-n.
static case_result compare_patched(ggml_type vtype, ggml_prec prec, int n,
                                   int queries, int rows) {
    case_result r;
    chain_out patched, built;
    if (!compute_chain(vtype, prec, REF_W, n, PATCH_FULL, queries, rows, patched)) return r;
    if (!compute_chain(vtype, prec, n,     n, PATCH_NONE, queries, rows, built))   return r;
    r.ran  = true;
    const bool p_same   = std::memcmp(patched.p.data(),   built.p.data(),   (size_t) n * queries * sizeof(float)) == 0;
    const bool out_same = std::memcmp(patched.out.data(), built.out.data(), (size_t) rows * queries * sizeof(float)) == 0;
    r.same = p_same && out_same;
    return r;
}

// OLD: two widths straddling a tinyBLAS tile class.
static case_result compare_tile_boundary(ggml_type vtype, ggml_prec prec, int a, int b,
                                         int queries, int rows) {
    case_result r;
    chain_out ca, cb;
    if (!compute_chain(vtype, prec, a, a, PATCH_NONE, queries, rows, ca)) return r;
    if (!compute_chain(vtype, prec, b, b, PATCH_NONE, queries, rows, cb)) return r;
    r.ran  = true;
    r.same = std::memcmp(ca.out.data(), cb.out.data(), (size_t) rows * queries * sizeof(float)) == 0;
    return r;
}

// The binary's own path, so the STRIDE child can be a FRESH process.
static const char * g_self = nullptr;


// STRIDE: incomplete patch. Returns true when the arm DETECTED a difference,
// either by computing something different or by aborting.
//
// The child MUST be a fresh process, not a fork of this one. This build links
// OpenMP (GGML_USE_OPENMP, -fopenmp), so ggml_graph_compute runs the graph
// inside `#pragma omp parallel`; libgomp is not fork-safe. A forked child
// inherits libgomp's bookkeeping without its worker threads, and its first
// GOMP_parallel deadlocks in gomp_team_start -- measured: the child sits in
// futex_do_wait under GOMP_parallel / ggml_graph_compute._omp_fn, the parent in
// do_wait, and the arm never returns. fork+exec starts the child with no
// inherited runtime state, which is the only correct way to isolate it.
//
// The child's exit code carries the verdict: 1 = computed something different,
// 0 = agreed, 2 = could not run. A signal or any other exit is a detection.
static bool stride_arm_detects(ggml_type vtype, ggml_prec prec, int n,
                               int queries, int rows) {
#if defined(__unix__)
    char a_n[16], a_q[16], a_r[16], a_v[16], a_p[16];
    std::snprintf(a_v, sizeof(a_v), "%d", (int) vtype);
    std::snprintf(a_p, sizeof(a_p), "%d", (int) prec);
    std::snprintf(a_n, sizeof(a_n), "%d", n);
    std::snprintf(a_q, sizeof(a_q), "%d", queries);
    std::snprintf(a_r, sizeof(a_r), "%d", rows);
    char * const argv[] = {
        (char *) g_self, (char *) "--stride-child", a_v, a_p, a_n, a_q, a_r, nullptr,
    };

    const pid_t pid = fork();
    if (pid < 0) return false; // cannot fork: report nothing rather than lie
    if (pid == 0) {
        execv(g_self, argv);
        _exit(2); // exec failed: could not run, not a detection
    }
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return false;
    if (WIFEXITED(status)) {
        const int code = WEXITSTATUS(status);
        if (code == 1) return true;   // computed something different
        if (code == 2) return false;  // could not run
        return true;                  // any other exit is a detection
    }
    return true; // signalled or aborted: detection
#else
    (void) vtype; (void) prec; (void) n; (void) queries; (void) rows;
    return false;
#endif
}

// The STRIDE child: one case, in a process with no inherited runtime state.
static int stride_child(int argc, char ** argv) {
    if (argc != 7) return 2;
    const ggml_type vtype = (ggml_type) std::atoi(argv[2]);
    const ggml_prec prec  = (ggml_prec)  std::atoi(argv[3]);
    const int n           = std::atoi(argv[4]);
    const int queries     = std::atoi(argv[5]);
    const int rows        = std::atoi(argv[6]);
    chain_out patched, built;
    if (!compute_chain(vtype, prec, REF_W, n, PATCH_NE_ONLY, queries, rows, patched) ||
        !compute_chain(vtype, prec, n,     n, PATCH_NONE,    queries, rows, built)) {
        return 2; // could not run: not a detection
    }
    const bool same = std::memcmp(patched.out.data(), built.out.data(),
                                  (size_t) rows * queries * sizeof(float)) == 0;
    return same ? 0 : 1;
}

int main(int argc, char ** argv) {
    g_self = argv[0];
    if (argc > 1 && std::strcmp(argv[1], "--stride-child") == 0) {
        return stride_child(argc, argv);
    }
    bool ok = true;

    // ---- STOCK: a complete patch must be an identity.
    int stock_agree = 0, stock_cases = 0;
    for (ggml_type vtype : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
        for (ggml_prec prec : { GGML_PREC_DEFAULT, GGML_PREC_F32_PEDANTIC }) {
            for (int qi = 0; qi < N_QUERIES; ++qi) {
                for (int ri = 0; ri < N_ROWS; ++ri) {
                    for (int i = 0; i < N_NARROW; ++i) {
                        const int n = NARROW_W[i];
                        char arm[64];
                        std::snprintf(arm, sizeof(arm), "stock_v%d_p%d_q%d_r%d",
                                      (int) vtype, (int) prec, QUERIES[qi], ROWS[ri]);
                        const case_result r = compare_patched(vtype, prec, n, QUERIES[qi], ROWS[ri]);
                        if (!r.ran) {
                            std::printf("KERNEL_SEL_FAIL arm=%s n=%d reason=compute\n", arm, n);
                            continue;
                        }
                        ++stock_cases;
                        if (r.same) ++stock_agree;
                        else {
                            std::printf("KERNEL_SEL_FAIL arm=%s n=%d reason=complete_patch_differs\n",
                                        arm, n);
                        }
                        std::printf("KERNEL_SEL_STOCK name=%s n=%d differ=%d\n",
                                    arm, n, r.same ? 0 : 1);
                        if (n == 1 || n == 16) emit_sites("stock", n, ROWS[ri]);
                    }
                }
            }
        }
    }
    if (stock_cases == 0 || stock_agree != stock_cases) {
        std::printf("KERNEL_SEL_FAIL arm=stock reason=complete_patch_not_identity "
                    "agree=%d/%d\n", stock_agree, stock_cases);
        ok = false;
    }

    // ---- STRIDE: an incomplete patch must be detected.
    int stride_detected = 0, stride_ran = 0;
    for (ggml_type vtype : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
        for (ggml_prec prec : { GGML_PREC_DEFAULT, GGML_PREC_F32_PEDANTIC }) {
            for (int qi = 0; qi < N_QUERIES; ++qi) {
                for (int ri = 0; ri < N_ROWS; ++ri) {
                    for (int i = 0; i < N_NARROW; ++i) {
                        const int n = NARROW_W[i];
                        char arm[64];
                        std::snprintf(arm, sizeof(arm), "stride_v%d_p%d_q%d_r%d",
                                      (int) vtype, (int) prec, QUERIES[qi], ROWS[ri]);
                        ++stride_ran;
                        const bool detected = stride_arm_detects(vtype, prec, n, QUERIES[qi], ROWS[ri]);
                        if (detected) ++stride_detected;
                        std::printf("KERNEL_SEL_STRIDE name=%s n=%d differ=%d reason=%s\n",
                                    arm, n, detected ? 1 : 0, detected ? "detected" : "none");
                    }
                }
            }
        }
    }
    if (stride_ran == 0 || stride_detected == 0) {
        std::printf("KERNEL_SEL_FAIL arm=stride reason=incomplete_patch_undetected "
                    "detected=%d/%d (the comparison cannot see a stride defect)\n",
                    stride_detected, stride_ran);
        ok = false;
    }

    // ---- OLD: the comparison must be able to see a selection difference.
    int old_differ = 0, old_ran = 0;
    static const int PAIRS[][2] = {
        {  7,   8 }, {  8,   9 }, { 15,  16 }, { 16,  17 },
        { 31,  32 }, { 32,  33 }, { 63,  64 }, { 64,  65 },
        {105, 106 }, {127, 128 }, {128, 129 }, {255, 256 },
    };
    for (ggml_type vtype : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
        for (ggml_prec prec : { GGML_PREC_DEFAULT, GGML_PREC_F32_PEDANTIC }) {
            for (int qi = 0; qi < N_QUERIES; ++qi) {
                for (int ri = 0; ri < N_ROWS; ++ri) {
                    for (const auto & pr : PAIRS) {
                        char arm[64];
                        std::snprintf(arm, sizeof(arm), "old_v%d_p%d_q%d_r%d",
                                      (int) vtype, (int) prec, QUERIES[qi], ROWS[ri]);
                        const case_result r = compare_tile_boundary(
                            vtype, prec, pr[0], pr[1], QUERIES[qi], ROWS[ri]);
                        if (!r.ran) {
                            std::printf("KERNEL_SEL_FAIL arm=%s n=%d n2=%d reason=compute\n",
                                        arm, pr[0], pr[1]);
                            continue;
                        }
                        ++old_ran;
                        if (!r.same) ++old_differ;
                        std::printf("KERNEL_SEL_OLD name=%s n=%d n2=%d differ=%d\n",
                                    arm, pr[0], pr[1], r.same ? 0 : 1);
                    }
                }
            }
        }
    }
    if (old_ran == 0 || old_differ == 0) {
        std::printf("KERNEL_SEL_FAIL arm=old reason=tile_boundary_agrees_everywhere "
                    "differ=%d/%d\n", old_differ, old_ran);
        ok = false;
    }

    std::printf("KERNEL_SEL_END ok=%d stock=%d/%d stride=%d/%d old=%d/%d\n",
                ok ? 1 : 0,
                stock_agree, stock_cases,
                stride_detected, stride_ran,
                old_differ, old_ran);
    return ok ? 0 : 1;
}
