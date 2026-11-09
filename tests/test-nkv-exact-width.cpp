#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

// Non-token n_kv exact-width gate.
//
// The token lane (LLAMA_EXPERIMENTAL_VARLEN_NTOK=1) is covered by
// test-ntok-strict. This test covers the OTHER lane: the default CPU path with
// the binder gate unset and the token lane off. There the graph is built once
// at the physical capacity and the binder patches the n_kv-coupled tensors in
// place to the true used span before every compute.
//
// THE ARM IS THE RUNNER'S. This process never sets, unsets or overrides
// GGML_TP5_UNION_BINDER, LLAMA_EXPERIMENTAL_VARLEN_NTOK or
// LLAMA_GRAPH_REUSE_DISABLE. It reads them once at start, prints them as one
// NKV_ARM line, and runs the case set that arm names. The three arms come from
// the runner (tests/nkv-cpu-probe/run-nkv-exact-width-arms-13022.sh, one
// process per arm with `env`) and from the ctest registrations
// (tests/CMakeLists.txt, p1-nkv-exact-*, `cmake -E env`):
//
//   unset : GGML_TP5_UNION_BINDER unset  -> the four binder cases run, `off`
//           is skipped (the qualified-CPU default, llama-context.cpp:556
//           via llama_union_binder_enabled)
//   off   : GGML_TP5_UNION_BINDER=0      -> `off` runs, the four binder cases
//           are skipped (stock control)
//   on    : GGML_TP5_UNION_BINDER=1      -> the four binder cases run, `off`
//           is skipped (the explicit opt-in branch)
//
// unset and on both exercise the binder, but through different branches of
// llama_context::llama_context; the NKV_ARM line makes that difference
// observable instead of assumed.
//
// Case x arm matrix. Every case is called in every arm; a case whose arm is
// not the running one prints skipped=1, touches nothing, and contributes
// neither a pass nor a failure. A skipped case is never a pass.
//
//   case          unset   on    off   why
//   single        run     run   skip  binder_ok() needs binder state
//   multi_seq     run     run   skip  same
//   multi_stream  run     run   skip  asserts the multi-stream bind refusal
//                                     (llama-context.cpp:4865) by the
//                                     diagnostics CONTENT: unset enabled=0,
//                                     on envelope_disabled=1/rebuild>=1;
//                                     patch_binds==0 on both. With the gate
//                                     explicitly off there is nothing to refuse
//   widths        run     run   skip  binder_ok() needs binder state
//   off           skip    skip  run   the stock control; its assertions are
//                                     the gate-off capacity pad, which only
//                                     holds with the gate explicitly off
//
// Observation contract. The eval callback must see all three n_kv runtime
// roots on every step. Two of them are nodes and one is an input, so the
// callback reaches them by two different routes:
//
//   union_nkv_k   llama_kv_cache::get_k  (src/llama-kv-cache.cpp:2184)
//   union_nkv_v   llama_kv_cache::get_v  (src/llama-kv-cache.cpp:2215, :2226)
//                 Both are ggml_view_4d results, i.e. GGML_OP_VIEW nodes, so
//                 the callback is called with them directly.
//
//   attn_inp_kq_mask  build_attn_inp_kq_mask (src/llama-graph.cpp:97, name at
//                     :111) and dsv4_build_raw_kq_mask (:976, name at :992)
//                 Both create a graph INPUT (ggml_new_tensor_4d +
//                 ggml_set_input), never a node. ggml_backend_sched_compute_splits
//                 (ggml/src/ggml-backend.cpp:2039) calls the eval callback over
//                 `split->graph.nodes` only, and an input travels in
//                 `split->inputs[]` (:1873), which the callback never sees.
//                 The mask is therefore observed on the nodes that CONSUME it
//                 (mask_of_node below): SOFT_MAX src[1], FLASH_ATTN_EXT and
//                 FLASH_ATTN_EXT_BANDED src[3], MUL_MAT src[2] when an
//                 attention flag is set. That is the same reachability the
//                 production binder uses (llama-context.cpp:4721).
//
// A root the callback never sees is a named failure, never a skip. The
// --negative mode drops one root from the callback's matching set on purpose,
// so the gate can prove that a missing observation fails. For the mask that
// means suppressing the consumer walk, not the name test.
//
// What this test proves, and what it does not:
//
//   * It DOES prove the executed width. The callback reads the mask's ne[0]
//     through its consumer, and the `union_nkv_k` / `union_nkv_v` views
//     directly, after the graph ran, and asserts their n_kv dimension equals
//     the 1-based position after the step. The callback runs after the binder
//     patch: the patch happens in process_ubatch before set_inputs and
//     compute, and the callback fires during compute.
//   * It DOES prove the reduction LANE the CPU kernels take. The binder
//     marks nothing on this lane (llama-context.cpp:5238-5252): it no longer
//     writes REDUCE_PAD_TAIL, MASKED_TAIL_GEMM or op_params[2], so there is
//     no pad-tail number to compare and the callback prints pad_tail=none.
//     The graph build selects the ordered kernels through ATTN_SOFT_MAX /
//     ATTN_MASK, and those flags are what this test asserts (see below).
//   * It DOES prove mask PROPERTIES for the multi-sequence cases. The mask is
//     a host buffer (set_input_kq_mask asserts a host buffer). It is also a
//     graph input: after its last consumer the allocator frees its block and
//     later nodes reuse it, so the callback COPIES the row data while the
//     input is valid (the callback is a needed node, so the scheduler
//     computes exactly it and calls back before any later node; the inputs
//     are filled before that loop). mask_row_props_ok reads the copy AFTER
//     llama_decode returned. Every column below the executed width must hold
//     exactly 0.0f or -INFINITY, and at least one column must be visible.
//     Which column is visible is NOT asserted: the physical cell index is
//     not exposed (find_slot allocates the lowest free cell, so after a
//     seq_rm cell index != decode index), and reconstructing it would be a
//     guess.
//   * It does NOT prove bitwise equality with the binder-off arm. The
//     binder-off arm still pads n_ctx/n_ctx_seq/n_ctx_kv to a 256-cell
//     bucket, so its CAPACITY differs and it rebuilds when the used span
//     moves; the executed width is the true used span on both arms (the
//     `off` case asserts exactly that). This test does not compare logits.
//
// The pad-tail width is an observation, not an assumption, and on this
// graph it is observably ABSENT. Read build_attn_mha
// (src/llama-graph.cpp:3299-3375) for this configuration:
//
//   * flash_attn_type is DISABLED, so the `else` branch runs;
//   * kq_mask is set and V is F16/F32, so :3371 calls
//     ggml_mul_mat_set_attn_mask(kqv, kq_mask). That function sets ATTN_MASK
//     on kqv and also ATTN_SOFT_MAX on kqv->src[1] (ggml.c:3403-3404), i.e.
//     on kq_soft_max;
//   * kq_soft_max is the only SOFT_MAX whose src[1] is attn_inp_kq_mask.
//
// The binder used to mark the softmax and the value product with
// REDUCE_PAD_TAIL / MASKED_TAIL_GEMM and pin op_params[2]. That emulation
// was removed (llama-context.cpp:5238-5252), so no node of this graph can
// carry the flag: the callback prints pad_tail=none on every step of every
// arm.
//
// What is asserted instead is the lane the graph build selected, in both
// directions:
//
//   * the mask-carrying softmax carries ATTN_SOFT_MAX, i.e. the ordered
//     reduction (ggml/src/ggml-cpu/ops.cpp:5527) that reads the mask and
//     skips masked columns by value (ggml/src/ggml-cpu/attn-ordered.cpp:61);
//   * that softmax does NOT carry REDUCE_PAD_TAIL, because the two lanes are
//     mutually exclusive (ggml/src/ggml.c:1339);
//   * the value product that consumes it carries ATTN_MASK, i.e. the ordered
//     product (ggml/src/ggml-cpu/ggml-cpu.c:1422-1430).
//
// The negative arm omit-pad-tail proves the check can fail. This graph has
// no node of its own for that root, so the ordered-lane evidence IS its
// observation: capture_shape() suppresses the two lane nodes when the root
// is omitted, exactly as it suppresses the mask's consumer walk for
// omit-mask. The case then fails with the named
//   FAIL: missing pad tail at step=0 (negative: omitted reduce_pad_tail ...)
// line. The stock control (`off`) has no binder and does not call this
// check; it prints pad_tail=none and is not asserted.
//
// Cases (one NKV_CASE header and one NKV_CASE_END footer per case):
//
//   single       binder unset or on, unified, 300 positions, 1 token per
//                step. The original Gate D case; it keeps the per-step
//                NKV_EXACT line.
//   multi_seq    binder unset or on, unified, n_seq_max=2, two sequences in one
//                unified stream. The executed width is the used CELL SPAN of
//                the shared stream (used_max_p1 over its cells), which public
//                seq_pos_max does NOT expose: two interleaved tokens at pos 0
//                occupy two cells (span 2, max position + 1 == 1), and after
//                a seq_rm find_slot can reuse a freed low cell so the span is
//                smaller than max position + 1. The case therefore asserts the
//                three observations agree, the observed width is in
//                [1, capacity], and the mask content; the position maximum is
//                printed as a hint only. The single case is the exact-width
//                proof.
//   multi_stream binder unset or on, non-unified, n_seq_max=2. The unset
//                gate does not qualify a multi-stream cache
//                (llama-context.cpp:556): the diagnostics slot may exist
//                (lazily allocated) but reports enabled=0 and no bind. The
//                on arm opens the gate; the multi-stream bind is refused
//                (llama-context.cpp:4865), which disables the envelope and
//                counts a rebuild fallback. The slot's CONTENT, not its
//                presence, is the witness. Both arms execute the true used
//                span, not the capacity: get_n_kv() rounds by n_pad (1 for a
//                plain cache). The capacity is arm-dependent; see
//                case_multi_stream. Every decode carries one token, so the
//                mask is [n_kv, 1, 1, 1] (llama-graph.cpp:87) and the current
//                token's row is row 0 (llama-kv-cache.cpp:2643).
//   widths       binder unset or on, unified, capacity 257. One sequence at
//                positions 0..256: the executed width crosses 1, 255, 256 and
//                257, and the last step runs at the full physical capacity.
//   off          binder explicitly off. The stock control: no binder state,
//                n_ctx padded to GGML_PAD(n_ctx, 256) (llama-context.cpp:669),
//                so the CAPACITY is 512. The executed width is the true used
//                span (n_pad is 1 for a plain cache), not the bucket; the
//                stock path builds at least once (OFF_ARM_BUILDS); it is not
//                the reuse arm.
//
// Not observable through the public API on CPU, and therefore NOT claimed:
//
//   * the `result > min_size` guard of get_n_kv (llama-kv-cache.cpp:2134).
//     With a unified cache n_stream is 1, so min_size is that stream's own
//     capacity and the width is min(capacity, used_max_p1) <= min_size by
//     construction. With a non-unified cache every stream has the same
//     capacity, and the binder refuses multi-stream before the guard can
//     matter. F32/F16 V has block size 1, so the align step cannot push the
//     width over min_size either. No public CPU input reaches the guard.
//   * the `v_trans` pad branch (align > 1). It needs a quantized V type whose
//     block size is > 1; the probe and this test use F32/F16 so the executed
//     width is the used span with no rounding. A quantized-V arm would prove
//     the pad, not the exact width, and is not part of this gate.
//   * a width that moves down through seq_rm alone. seq_rm frees the cells it
//     removes and the stream head then reuses the lowest free cell, so the
//     used span does not shrink. multi_seq moves the width down only by
//     removing a whole sequence from the shared stream.

// Usage:
//   test-nkv-exact-width model.gguf
//   test-nkv-exact-width model.gguf --negative omit-mask|omit-k|omit-v|omit-pad-tail
//
// --negative keeps the arm and the case set and removes exactly one root from
// the callback's matching set, so the case that runs must fail with the named
// line for that root. The process then exits with one of three distinct
// statuses, and the two non-zero ones are different verdicts:
//
//   0  the omission was DETECTED: the run did not pass, a case recorded the
//      missing observation, and the record names the root this run was told
//      to omit. The last line is
//          NKV_NEGATIVE ok=1 omitted=<root-name>
//      where the root name is one of
//          attn_inp_kq_mask union_nkv_k union_nkv_v reduce_pad_tail
//      and the named line is
//          FAIL: missing tensor at step=0 mask=.. k=.. v=.. (negative: ...)
//      for the three n_kv roots, or
//          FAIL: missing pad tail at step=0 (negative: omitted reduce_pad_tail ...)
//      for omit-pad-tail. ctest reports PASS for this exit, so the
//      registration in tests/CMakeLists.txt pins the reason with
//      PASS_REGULAR_EXPRESSION on the NKV_NEGATIVE line (one per root) and
//      FAIL_REGULAR_EXPRESSION on the "negative run ... passed" text.
//   1  the omission was NOT proven: the run passed (the gate cannot see a
//      missing observation), or it failed without recording the named
//      missing observation (model load, context failure, wrong arm). The
//      FAIL: line says which.
//   2  usage error (bad --negative target, unknown arm value).
//
// omit-pad-tail is the negative arm for the pad-tail width. It suppresses the
// pad-tail observation only; the mask, K and V roots are observed as usual,
// so the case fails on the missing pad tail and on nothing else.
//
// Exit 1 is never "the negative run worked": only exit 0 plus the
// NKV_NEGATIVE line is. A gate that keys on the exit code alone would accept
// a run that failed for an unrelated reason, which is why the gate script and
// the ctest registrations both key on the line.

namespace {

constexpr int steps_single = 300;
constexpr int n_ctx_single = 300; // deliberately not a multiple of 256
constexpr int n_ctx_widths = 257; // crosses 255/256/257 and ends at capacity
constexpr int width_max    = 4;

// The binder arm this process runs, set once in main() from the runner's
// environment and never written again. The cases read it to decide whether
// they are valid here; none of them changes it.
enum class binder_arm : int {
    unset = 0,
    off   = 1,
    on    = 2,
};

binder_arm g_arm = binder_arm::unset;

// machine-readable case accounting; a shell gate reads the final line
int  g_cases_run = 0;
int  g_steps_run = 0;
bool g_all_ok    = true;

// One n_kv runtime root the callback can observe.
enum class nkv_root : int {
    mask     = 0,
    k        = 1,
    v        = 2,
    pad_tail = 3,
};

const char * nkv_root_name(nkv_root root) {
    switch (root) {
        case nkv_root::mask:     return "attn_inp_kq_mask";
        case nkv_root::k:        return "union_nkv_k";
        case nkv_root::v:        return "union_nkv_v";
        case nkv_root::pad_tail: return "reduce_pad_tail";
    }
    return "?";
}

// The root a --negative run removes from the matching set. -1 = none.
int g_omit_root = -1;

// What a --negative run must prove, recorded only when the observation for
// the omitted root is actually missing. A nonzero exit is NOT proof: a model
// load failure, a context failure or a wrong arm also exit nonzero, and none
// of them shows that the gate can tell a missing observation from a present
// one. main() reads these fields, and nothing else, to decide whether the
// negative run succeeded.
struct negative_evidence {
    bool     missing_seen  = false; // the named "missing tensor" line was printed
    nkv_root missing_root  = nkv_root::mask;
    bool     pad_tail_seen = false; // the named "missing pad tail" line was printed
    int      pad_tail_step = -1;
};

negative_evidence g_negative;

// The omitted root is the one whose observation is missing. Record the
// evidence and print the named line. Called only from a case, at the point
// where that case would otherwise have failed for the missing observation.
void note_missing_observation(int step, const char * case_name) {
    const nkv_root root = (nkv_root) g_omit_root;
    g_negative.missing_seen = true;
    g_negative.missing_root = root;
    std::fprintf(stderr,
                 "FAIL: missing tensor at step=%d mask=%d k=%d v=%d "
                 "(negative: omitted %s in case %s)\n",
                 step,
                 int(root == nkv_root::mask),
                 int(root == nkv_root::k),
                 int(root == nkv_root::v),
                 nkv_root_name(root), case_name);
}

// The pad tail was omitted and is therefore never recorded. Same contract as
// note_missing_observation(): the evidence and the named line are printed
// together, so a log carrying NKV_NEGATIVE ok=1 always carries this line.
void note_missing_pad_tail(int step, const char * case_name) {
    g_negative.pad_tail_seen = true;
    g_negative.pad_tail_step = step;
    std::fprintf(stderr,
                 "FAIL: missing pad tail at step=%d (negative: omitted %s in case %s)\n",
                 step, nkv_root_name(nkv_root::pad_tail), case_name);
}

struct shape_capture {
    bool seen_mask = false;
    bool seen_k    = false;
    bool seen_v    = false;

    int64_t       mask_ne0  = -1;
    int64_t       mask_ne1  = -1;
    int64_t       mask_ne3  = -1;
    int64_t       k_nkv     = -1;
    int64_t       v_nkv     = -1;
    enum ggml_type mask_type = GGML_TYPE_COUNT;

    // The mask content, copied inside the eval callback. The mask is a graph
    // input: after its last consumer the allocator frees its block and later
    // nodes reuse it, so reading mask->data after llama_decode returns reads
    // another tensor's data. The copy is F32-only; an F16 mask is reported by
    // mask_row_props_ok as a named failure.
    std::vector<float> mask_data;       // rows * mask_ne0 floats, row-major
    int64_t            mask_data_rows   = 0;
    bool               mask_data_copied = false;

    // The pad-tail width of the consumer node, and whether that node carries
    // GGML_TENSOR_FLAG_REDUCE_PAD_TAIL at all. pad_tail_seen is false when no
    // node in the graph carries the flag: the line then prints pad_tail=none.
    // On this graph that is the expected state (see the file header), so the
    // pad tail is printed as evidence and the lane is asserted separately.
    bool    pad_tail_seen = false;
    int32_t pad_tail      = -1;

    // The two nodes of the attention lane the walk visits: the mask-carrying
    // softmax (SOFT_MAX src[1]) and the value product (MUL_MAT src[2]). Their
    // flags are what tells the ordered lane from the padded one.
    bool    seen_softmax   = false;
    bool    seen_value_mat = false;
    int32_t softmax_flags  = 0;
    int32_t value_flags    = 0;

    void reset() {
        seen_mask      = false;
        seen_k         = false;
        seen_v         = false;
        mask_ne0       = -1;
        mask_ne1       = -1;
        mask_ne3       = -1;
        k_nkv          = -1;
        v_nkv          = -1;
        mask_type      = GGML_TYPE_COUNT;
        mask_data.clear();
        mask_data_rows   = 0;
        mask_data_copied = false;
        pad_tail_seen  = false;
        pad_tail       = -1;
        seen_softmax   = false;
        seen_value_mat = false;
        softmax_flags  = 0;
        value_flags    = 0;
    }
};

// The n_kv dimension of the V view depends on the cache layout: get_v builds
// ggml_view_4d(head_v_eff, n_head_kv, n_kv, ns) when !v_trans and
// ggml_view_4d(n_kv, n_head_kv, head_v_eff, ns) when v_trans. The two cases
// differ in which stride is the largest, so read the layout rather than assume.
int64_t v_nkv_dim(const ggml_tensor * t) {
    return t->nb[1] > t->nb[2] ? t->ne[0] : t->ne[2];
}

// Resolve a tensor to one of the three roots by the name its producer sets
// (see the file header). A name no producer sets, including the empty name of
// an unnamed tensor, resolves to nothing and is never counted as observed.
bool nkv_root_of(const char * name, nkv_root & out) {
    if (strcmp(name, "attn_inp_kq_mask") == 0) {
        out = nkv_root::mask;
        return true;
    }
    if (strcmp(name, "union_nkv_k") == 0) {
        out = nkv_root::k;
        return true;
    }
    if (strcmp(name, "union_nkv_v") == 0) {
        out = nkv_root::v;
        return true;
    }
    return false;
}

// The mask is a graph INPUT, not a node, so the eval callback is never called
// with it. ggml_backend_sched_compute_splits (ggml-backend.cpp:2039) walks
// `split->graph.nodes` only; an input appears in `split->inputs[]` (:1873),
// which the callback never sees. Naming the mask cannot change that.
//
// So the mask is observed where it is reachable: on the nodes that CONSUME
// it. Every consumer of the generic mask keeps it in a fixed src slot:
//
//   GGML_OP_SOFT_MAX              src[1]  ggml_soft_max_impl          (ggml.c:4183)
//   GGML_OP_FLASH_ATTN_EXT        src[3]  ggml_flash_attn_ext         (ggml.c:5550)
//   GGML_OP_FLASH_ATTN_EXT_BANDED src[3]                              (ggml.c:5597)
//   GGML_OP_MUL_MAT + attn flags  src[2]  ggml_mul_mat_set_attn_*     (ggml.c:3394,3408)
//
// This is the same reachability the production binder uses:
// llama-context.cpp:4721 checks `x->src[1]->name` on a SOFT_MAX node.
//
// The resolver reads the SHAPE only. The callback copies the row data on its
// ask=false call (see capture_shape): the scheduler has already set the graph
// inputs, and the callback for a mask consumer is a "needed" node, so the
// copy is taken before any later node can reuse the input's storage. The
// content check then runs on that copy after llama_decode returned.
struct mask_slot {
    enum ggml_op op;
    int          src;
};

constexpr mask_slot mask_slots[] = {
    { GGML_OP_SOFT_MAX,               1 },
    { GGML_OP_FLASH_ATTN_EXT,         3 },
    { GGML_OP_FLASH_ATTN_EXT_BANDED,  3 },
    { GGML_OP_MUL_MAT,                2 },
};

// The consumer node the walk stands on, or nullptr. It is the node whose
// mask slot resolved in mask_of_node(); the pad-tail width is read from it,
// not from the mask, so the two numbers come from different write paths.
ggml_tensor * consumer_of_node(ggml_tensor * t) {
    if (t == nullptr) {
        return nullptr;
    }
    for (const mask_slot & slot : mask_slots) {
        if (t->op != slot.op) {
            continue;
        }
        if (t->op == GGML_OP_MUL_MAT &&
                (t->flags & (GGML_TENSOR_FLAG_ATTN_MASK | GGML_TENSOR_FLAG_ATTN_SCORES)) == 0) {
            continue;
        }
        ggml_tensor * m = t->src[slot.src];
        if (m != nullptr && strcmp(ggml_get_name(m), "attn_inp_kq_mask") == 0) {
            return t;
        }
    }
    return nullptr;
}

// The mask a node consumes, or nullptr. A MUL_MAT carries a mask in src[2]
// only when one of the attention flags is set; without them src[2] is unused.
ggml_tensor * mask_of_node(ggml_tensor * t) {
    ggml_tensor * consumer = consumer_of_node(t);
    if (consumer == nullptr) {
        return nullptr;
    }
    for (const mask_slot & slot : mask_slots) {
        if (consumer->op == slot.op && consumer->src[slot.src] != nullptr &&
                strcmp(ggml_get_name(consumer->src[slot.src]), "attn_inp_kq_mask") == 0) {
            return consumer->src[slot.src];
        }
    }
    return nullptr;
}

// The pad-tail width the CPU kernel reads for this node, or -1 when the node
// does not carry GGML_TENSOR_FLAG_REDUCE_PAD_TAIL. ggml_get_op_params_i32 is
// static in ggml-impl.h, so the public struct member is read directly, the
// same int32_t slot the consumer would read through
// ggml_get_op_params_i32(dst, 2) (ops.cpp:5597). No producer on this lane
// writes it (llama-context.cpp:5238-5252), so the value is only read when a
// node actually carries the flag.
int32_t pad_tail_of_node(const ggml_tensor * t) {
    if (t == nullptr || (t->flags & GGML_TENSOR_FLAG_REDUCE_PAD_TAIL) == 0) {
        return -1;
    }
    return t->op_params[2];
}

bool capture_shape(ggml_tensor * t, bool ask, void * user_data) {
    nkv_root root = nkv_root::mask;
    const bool named = nkv_root_of(ggml_get_name(t), root);
    // the mask is only reachable through a consumer, never as a node
    ggml_tensor * consumer = named ? nullptr : consumer_of_node(t);
    ggml_tensor * mask     = consumer != nullptr ? mask_of_node(consumer) : nullptr;
    // the pad tail is observed on its own, independent of the mask walk: the
    // node that carries it is the value product, whose src[1] is the softmax
    const int32_t pad_tail = pad_tail_of_node(t);
    // --negative: the omitted root is invisible to this callback, exactly as
    // it would be if its producer stopped setting the name. For the mask that
    // means the consumer walk is suppressed, because the mask has no node of
    // its own to suppress. For the pad tail it means the width is never
    // recorded; and because no node on this graph carries the flag, the
    // ordered-lane evidence below is the observation of that root and is
    // suppressed too.
    const bool want_pad_tail = pad_tail >= 0 && g_omit_root != int(nkv_root::pad_tail);
    const bool wanted = (named && int(root) != g_omit_root) ||
                        (mask != nullptr && g_omit_root != int(nkv_root::mask)) ||
                        want_pad_tail;

    if (ask) {
        return wanted;
    }
    if (!wanted) {
        return true;
    }

    auto & out = *static_cast<shape_capture *>(user_data);

    if (want_pad_tail) {
        out.pad_tail_seen = true;
        out.pad_tail      = pad_tail;
    }
    // the lane the mask is consumed on: the mask-carrying softmax and the
    // value product, i.e. the MUL_MAT that carries ATTN_MASK. kq itself also
    // consumes the mask but carries ATTN_SCORES, so it is not the value
    // product and must not overwrite this evidence.
    //
    // --negative omit-pad-tail: this graph has no node that carries
    // REDUCE_PAD_TAIL, so the ordered-lane evidence IS the observation of
    // that root. Suppress it exactly as the mask walk is suppressed for
    // omit-mask. The mask observation itself stays, so the case reaches the
    // pad-tail check instead of failing as a missing tensor.
    if (mask != nullptr && g_omit_root != int(nkv_root::pad_tail)) {
        if (t->op == GGML_OP_SOFT_MAX) {
            out.seen_softmax  = true;
            out.softmax_flags = t->flags;
        } else if (t->op == GGML_OP_MUL_MAT &&
                   (t->flags & GGML_TENSOR_FLAG_ATTN_MASK) != 0) {
            out.seen_value_mat = true;
            out.value_flags    = t->flags;
        }
    }

    if (mask != nullptr) {
        out.seen_mask = true;
        out.mask_ne0  = mask->ne[0];
        out.mask_ne1  = mask->ne[1];
        out.mask_ne3  = mask->ne[3];
        out.mask_type = mask->type;
        // Copy the content while the input is valid. This callback is the
        // mask's consumer, and capture_shape() returns true for it, so the
        // scheduler computes exactly this node and calls us back before any
        // later node runs (ggml-backend.cpp:2043-2063). The input was filled
        // by set_input_kq_mask before the compute loop: the input copies are
        // flushed at :2029, before the callback loop starts. The
        // allocator does not free a parent before allocating the node that
        // consumes it (ggml-alloc.c:768-819), so the buffer still holds the
        // fill here. After llama_decode returns the block may have been
        // reused by a later node, which is why the copy is taken now.
        if (!out.mask_data_copied) {
            out.mask_data_copied = true;
            const int64_t ne0        = mask->ne[0];
            const int64_t rows_total = mask->ne[1]*mask->ne[3];
            const int64_t max_floats = 4*512; // sane bound: 2048 floats
            const int64_t rows_fit   = ne0 > 0 ? max_floats/ne0 : 0;
            const int64_t rows       = rows_total < rows_fit ? rows_total : rows_fit;
            if (mask->type == GGML_TYPE_F32 && mask->data != nullptr && rows > 0) {
                const float * src = (const float *) mask->data;
                out.mask_data.assign(src, src + rows*ne0);
                out.mask_data_rows = rows;
            }
        }
        return true;
    }

    switch (root) {
        case nkv_root::mask:
            // unreachable: the mask is never a node
            break;
        case nkv_root::k:
            // get_k: ggml_view_4d(head_k_eff, n_head_kv, n_kv, ns)
            out.seen_k = true;
            out.k_nkv  = t->ne[2];
            break;
        case nkv_root::v:
            out.seen_v = true;
            out.v_nkv  = v_nkv_dim(t);
            break;
    }

    return true;
}

// Mask properties that need no cell map.
//
// The mask is a host F32 buffer (llama-kv-cache.cpp set_input_kq_mask asserts
// ggml_backend_buffer_is_host), filled by set_input_kq_mask_impl. The physical
// cell index of a token is NOT derivable from the public API: find_slot
// allocates the lowest free cell, so after a seq_rm the next token reuses a
// freed cell and cell index != decode index. This helper therefore asserts
// only two properties, neither of which names a column:
//
//   * every column j < width holds exactly 0.0f or -INFINITY (the fill loop
//     writes n_kv columns and nothing else, so any other value means the row
//     was not written by this step);
//   * at least one column is visible. WHICH column is not asserted: the
//     query's own cell must be visible, but its index is not exposed, so a
//     mask that hides the query's own cell while keeping some other cell
//     visible would pass. That is the limit of a property check.
//
// Columns >= width are not read: the fill loop never writes them.
//
// Reads the COPY of the mask taken inside the eval callback, not the live
// tensor. The mask is a graph input: after its last consumer the allocator
// frees its block and later nodes reuse it, so mask->data read after
// llama_decode returns is not the mask any more.
//
// The mask is F16 when flash attention is on and F32 otherwise
// (build_attn_inp_kq_mask picks by cparams.flash_attn). This test disables
// flash attention, so the expected type is F32; the type is asserted rather
// than assumed, and an F16 mask is reported instead of read as float.
bool mask_row_props_ok(const shape_capture & cap, int query_pos, int width,
                       int64_t row) {
    if (cap.mask_type != GGML_TYPE_F32) {
        std::fprintf(stderr, "FAIL: mask type=%d, want F32 (%d)\n",
                     int(cap.mask_type), int(GGML_TYPE_F32));
        return false;
    }
    if (!cap.mask_data_copied) {
        std::fprintf(stderr,
                "FAIL: mask content was not captured at query pos %d "
                "(the eval callback must copy it while the input is valid)\n",
                query_pos);
        return false;
    }
    if (cap.mask_data_rows <= 0 || cap.mask_ne0 <= 0) {
        std::fprintf(stderr,
                "FAIL: mask content copy is empty (rows=%lld ne0=%lld)\n",
                (long long) cap.mask_data_rows, (long long) cap.mask_ne0);
        return false;
    }
    if (width <= 0 || (int64_t) width > cap.mask_ne0) {
        std::fprintf(stderr, "FAIL: width=%d outside [1, mask ne0=%lld]\n",
                     width, (long long) cap.mask_ne0);
        return false;
    }
    if (row < 0 || row >= cap.mask_data_rows) {
        std::fprintf(stderr, "FAIL: mask row %lld outside copied rows [0, %lld)\n",
                     (long long) row, (long long) cap.mask_data_rows);
        return false;
    }
    const float * r = cap.mask_data.data() + row*cap.mask_ne0;
    int visible = 0;
    for (int j = 0; j < width; ++j) {
        const bool keep = r[j] == 0.0f;
        const bool drop = r[j] == -INFINITY;
        if (!keep && !drop) {
            std::fprintf(stderr,
                    "FAIL: mask column %d is neither 0 nor -inf (%f)\n",
                    j, (double) r[j]);
            return false;
        }
        if (keep) {
            ++visible;
        }
    }
    if (visible == 0) {
        std::fprintf(stderr,
                "FAIL: no visible column for query pos %d at width %d "
                "(at least one column must be visible)\n", query_pos, width);
        return false;
    }
    return true;
}



int token_at(int pos, int vocab) {
    return 3 + ((pos * 7 + 1) % (vocab - 3));
}

// One decode of a single token. Returns the llama_decode status.
int decode_one(llama_context * ctx, int token, int pos, int seq) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    if (!llama_batch_is_complete(&batch, 1, false, true)) {
        llama_batch_free(batch);
        batch = llama_batch{};
        return -100;
    }
    batch.n_tokens     = 1;
    batch.token[0]     = token;
    batch.pos[0]       = pos;
    batch.n_seq_id[0]  = 1;
    batch.seq_id[0][0] = seq;
    batch.logits[0]    = 1;
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc;
}

llama_context * make_ctx(llama_model * model, int n_ctx, int n_ctx_kv, int n_seq_max,
                         bool unified, shape_capture * cap) {
    auto cp = llama_context_default_params();
    cp.n_ctx         = n_ctx;
    cp.n_ctx_kv      = n_ctx_kv;
    cp.n_batch       = width_max;
    cp.n_ubatch      = width_max;
    cp.n_outputs_max = width_max;
    cp.n_seq_max     = n_seq_max;
    cp.n_threads     = cp.n_threads_batch = 1;
    cp.kv_unified    = unified;
    cp.offload_kqv   = cp.op_offload = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.no_perf       = false;
    cp.cb_eval           = cap != nullptr ? capture_shape : nullptr;
    cp.cb_eval_user_data = cap;
    return llama_init_from_model(model, cp);
}

// case header/footer: a shell gate keys on these and never guesses.
//
// Every case names the arm it is valid in. A case whose arm is not the one
// this process runs is reported as skipped=1 and is NOT counted: a skip is
// visible in the log and never becomes a pass.
void case_begin(const char * name, int n_seq, bool unified, int steps,
                const char * arm, bool skipped) {
    std::fprintf(stderr, "NKV_CASE name=%s n_seq=%d unified=%d steps=%d arm=%s skipped=%d\n",
                 name, n_seq, int(unified), steps, arm, int(skipped));
}

void case_end(const char * name, bool ok, int steps, bool skipped) {
    if (!skipped) {
        ++g_cases_run;
        g_steps_run += steps;
        if (!ok) {
            g_all_ok = false;
        }
    }
    std::fprintf(stderr, "NKV_CASE_END name=%s ok=%d steps=%d skipped=%d\n",
                 name, int(ok), steps, int(skipped));
}

// the binder state a passing single-stream case must show
bool binder_ok(llama_context * ctx, int step) {
    llama_union_diagnostics d = {};
    if (!llama_get_union_diagnostics(ctx, &d)) {
        std::fprintf(stderr, "FAIL: no binder diagnostics at step=%d\n", step);
        return false;
    }
    if (!d.enabled || d.envelope_disabled || !d.record_valid || d.token_map_valid ||
            d.rebuild_fallbacks != 0) {
        std::fprintf(stderr,
                "FAIL: binder state at step=%d enabled=%d envelope_disabled=%d "
                "record_valid=%d token_map_valid=%d rebuild_fallbacks=%llu\n",
                step, int(d.enabled), int(d.envelope_disabled), int(d.record_valid),
                int(d.token_map_valid), (unsigned long long) d.rebuild_fallbacks);
        return false;
    }
    // the token lane is off, so llama_get_ntok_plan_stats must refuse
    llama_ntok_plan_stats stats = {};
    if (llama_get_ntok_plan_stats(ctx, &stats)) {
        std::fprintf(stderr, "FAIL: token lane active at step=%d\n", step);
        return false;
    }
    return true;
}

// The pad-tail field as the log prints it: pad_tail=<n> when a node carried
// GGML_TENSOR_FLAG_REDUCE_PAD_TAIL, pad_tail=none when none did. The token is
// the same in every line family so a reader can grep one pattern.
const char * pad_tail_text(const shape_capture & cap) {
    static char buf[16];
    if (!cap.pad_tail_seen) {
        return "none";
    }
    std::snprintf(buf, sizeof(buf), "%d", (int) cap.pad_tail);
    return buf;
}

// The attention lane the mask is consumed on, asserted in both directions.
//
// This graph does not carry REDUCE_PAD_TAIL: the binder marks nothing on this
// lane (llama-context.cpp:5238-5252), so no node can carry the flag. The
// ordered lane is selected by the graph build itself
// (ggml_mul_mat_set_attn_mask sets ATTN_MASK on the value product and
// ATTN_SOFT_MAX on its softmax, ggml.c:3403-3404). The honest check is the
// one below: the ordered lane must be the lane this graph is on, and the
// padded lane must not be half-set.
//
//   * the softmax carries ATTN_SOFT_MAX (ordered reduction, ops.cpp:5527);
//   * the softmax does NOT carry REDUCE_PAD_TAIL (mutually exclusive,
//     ggml.c:1339);
//   * the value product carries ATTN_MASK (ordered product, ggml-cpu.c:1422).
//
// A pad tail that IS present is checked against the bound width: if a future
// graph change puts this lane on the padded path, the number must still equal
// the width the bind set. Its absence is not a pass by itself: the
// ordered-lane evidence above must be present, because on this graph that
// evidence is the only observation of the pad-tail root.
//
// --negative omit-pad-tail suppresses exactly that evidence in
// capture_shape(), so the missing branch below is what records the negative.
bool pad_tail_lane_ok(const shape_capture & cap, int64_t want, const char * case_name,
                      int step) {
    if (cap.pad_tail_seen) {
        if ((int64_t) cap.pad_tail != want) {
            std::fprintf(stderr,
                    "FAIL: pad tail at step=%d want=%lld got=%d\n",
                    step, (long long) want, (int) cap.pad_tail);
            return false;
        }
        if (!cap.seen_softmax || !cap.seen_value_mat) {
            std::fprintf(stderr,
                    "FAIL: lane nodes at step=%d softmax=%d value_mat=%d\n",
                    step, int(cap.seen_softmax), int(cap.seen_value_mat));
            return false;
        }
    } else if (!cap.seen_softmax || !cap.seen_value_mat) {
        if (g_omit_root == int(nkv_root::pad_tail)) {
            note_missing_pad_tail(step, case_name);
        }
        std::fprintf(stderr,
                "FAIL: missing pad tail at step=%d want=%lld (no node carries "
                "REDUCE_PAD_TAIL and the ordered-lane evidence is absent: "
                "softmax=%d value_mat=%d)\n",
                step, (long long) want, int(cap.seen_softmax),
                int(cap.seen_value_mat));
        return false;
    }
    if ((cap.softmax_flags & GGML_TENSOR_FLAG_ATTN_SOFT_MAX) == 0) {
        std::fprintf(stderr,
                "FAIL: softmax flags=0x%x lack ATTN_SOFT_MAX at step=%d\n",
                (unsigned) cap.softmax_flags, step);
        return false;
    }
    if ((cap.softmax_flags & GGML_TENSOR_FLAG_REDUCE_PAD_TAIL) != 0) {
        std::fprintf(stderr,
                "FAIL: softmax flags=0x%x carry REDUCE_PAD_TAIL at step=%d "
                "(the two lanes are mutually exclusive)\n",
                (unsigned) cap.softmax_flags, step);
        return false;
    }
    if ((cap.value_flags & GGML_TENSOR_FLAG_ATTN_MASK) == 0) {
        std::fprintf(stderr,
                "FAIL: value product flags=0x%x lack ATTN_MASK at step=%d\n",
                (unsigned) cap.value_flags, step);
        return false;
    }
    return true;
}

// The binder's own exported width, after the decode returned. This is a
// THIRD binder-side number, not kernel evidence: runtime_n_kv is the value
// written into the graph's n_kv input, bound_n_kv the width the last applied
// bind set (include/llama.h:689-697, src/llama-context.cpp:1900-1912). It is
// printed so the log carries it next to the two shape numbers, and it is
// labelled binder-side wherever it appears.
//
// patch_binds is the binder's count of applied binds so far. It is printed
// because the mask shape is written inside a bind (src/llama-context.cpp,
// the shape-map patch), so a step that observed it at a new width must show
// the count moved. The binder marks no reduction-order flag at all
// (src/llama-context.cpp:5238-5252). A step whose bind was refused keeps the
// previous patch and is reported by binder_ok() (rebuild_fallbacks); the
// count is the ordering evidence, not a width.
bool nkv_runtime_ok(llama_context * ctx, int64_t want, int step) {
    llama_nkv_runtime_stats st = {};
    if (!llama_get_nkv_runtime_stats(ctx, &st)) {
        std::fprintf(stderr, "FAIL: no n_kv runtime stats at step=%d\n", step);
        return false;
    }
    llama_union_diagnostics d = {};
    const bool has_d = llama_get_union_diagnostics(ctx, &d);
    std::fprintf(stderr,
                 "NKV_RT step=%d runtime=%u bound=%u pos=%lld patch_binds=%llu\n",
                 step, st.runtime_n_kv, st.bound_n_kv, (long long) want,
                 (unsigned long long) (has_d ? d.patch_binds : 0));
    if ((int64_t) st.runtime_n_kv != want || (int64_t) st.bound_n_kv != want) {
        std::fprintf(stderr,
                "FAIL: n_kv runtime stats at step=%d runtime=%u bound=%u want=%lld\n",
                step, st.runtime_n_kv, st.bound_n_kv, (long long) want);
        return false;
    }
    return true;
}

// Case: binder unset, unified, one sequence, 1 token per step.
bool case_single(llama_model * model, int vocab) {
    const int steps = steps_single;
    // valid in the unset and on arms (both run the binder); the off arm
    // reports it as skipped
    const bool skipped = g_arm == binder_arm::off;
    case_begin("single", 1, true, steps, "binder", skipped);
    if (skipped) {
        // not this arm's case: report the skip and touch nothing
        case_end("single", true, 0, true);
        return true;
    }

    shape_capture cap;
    llama_context * ctx = make_ctx(model, n_ctx_single, n_ctx_single, 1, true, &cap);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: context (single case)\n");
        case_end("single", false, 0, skipped);
        return false;
    }

    bool ok = true;

    // the capacity sites are not 256-padded on this lane
    const uint32_t got_ctx     = llama_n_ctx(ctx);
    const uint32_t got_ctx_seq = llama_n_ctx_seq(ctx);
    const uint32_t got_ctx_kv  = llama_n_ctx_kv(ctx);
    if (got_ctx != (uint32_t) n_ctx_single || got_ctx_seq != (uint32_t) n_ctx_single ||
            got_ctx_kv != (uint32_t) n_ctx_single) {
        std::fprintf(stderr, "FAIL: ctx n_ctx=%u n_ctx_seq=%u n_ctx_kv=%u want=%d\n",
                     got_ctx, got_ctx_seq, got_ctx_kv, n_ctx_single);
        ok = false;
    }

    const uint64_t builds_init = llama_graph_build_count(ctx);

    int pos = 0;
    for (int step = 0; ok && step < steps; ++step) {
        cap.reset();
        const int rc = decode_one(ctx, token_at(pos, vocab), pos, 0);
        if (rc != 0) {
            std::fprintf(stderr, "FAIL: decode step=%d rc=%d\n", step, rc);
            ok = false;
            break;
        }
        pos += 1;

        if (!cap.seen_mask || !cap.seen_k || !cap.seen_v) {
            if (g_omit_root >= 0) {
                note_missing_observation(step, "single");
            }
            std::fprintf(stderr,
                    "FAIL: missing tensor at step=%d mask=%d k=%d v=%d "
                    "(the shape callback must observe attn_inp_kq_mask, "
                    "union_nkv_k and union_nkv_v)\n",
                    step, int(cap.seen_mask), int(cap.seen_k), int(cap.seen_v));
            ok = false;
            break;
        }
        std::fprintf(stderr,
                "NKV_EXACT step=%d pos=%d mask_ne0=%lld k_ne2=%lld v_nkv=%lld "
                "pad_tail=%s\n",
                step, pos,
                (long long) cap.mask_ne0, (long long) cap.k_nkv, (long long) cap.v_nkv,
                pad_tail_text(cap));
        if (cap.mask_ne0 != pos || cap.k_nkv != pos || cap.v_nkv != pos) {
            std::fprintf(stderr,
                    "FAIL: padded width at step=%d pos=%d mask=%lld k=%lld v=%lld\n",
                    step, pos, (long long) cap.mask_ne0,
                    (long long) cap.k_nkv, (long long) cap.v_nkv);
            ok = false;
            break;
        }
        if (!pad_tail_lane_ok(cap, pos, "single", step)) {
            ok = false;
            break;
        }
        if (!nkv_runtime_ok(ctx, pos, step)) {
            ok = false;
            break;
        }
        if (!binder_ok(ctx, step)) {
            ok = false;
            break;
        }
    }

    // one graph build for the whole run: every width change was a bind
    const uint64_t builds_total = llama_graph_build_count(ctx);
    if (ok && builds_total - builds_init != 1) {
        std::fprintf(stderr, "FAIL: graph builds=%llu want=1\n",
                     (unsigned long long) (builds_total - builds_init));
        ok = false;
    }
    if (ok && pos != steps) {
        std::fprintf(stderr, "FAIL: pos=%d want=%d\n", pos, steps);
        ok = false;
    }

    llama_free(ctx);
    case_end("single", ok, steps, skipped);
    return ok;
}

// Case: binder unset or on, unified, two sequences sharing one stream.
//
// Both sequences live in the one unified stream, so get_n_kv reports one
// shared width. The width is the POSITION span of that stream, rounded by
// n_pad and clamped to the cell count:
//   width = min(cells.size(), max(n_pad, GGML_PAD(cells.used_pos_span_p1(), n_pad)))
// (llama_kv_cache::get_n_kv), where used_pos_span_p1() is max(position)+1 over
// the stream's used cells (llama_kv_cells::used_pos_span_p1).
//
// A cell index is not a position. A unified cache holds one cell per position,
// and two sequences at one position share that cell: find_slot reuses the cell
// that already holds the position (llama_kv_cells::find_cell_by_pos). The
// width therefore covers positions, not cells, and the public position maximum
// is exact. This case asserts it.
//
// This case asserts:
//
//   (a) the three width observations agree with the position span:
//       mask ne0 == K n_kv == V n_kv == max(seq_pos_max+1), clamped to the KV
//       capacity;
//   (b) every column j < width is 0.0f or -INFINITY (columns >= width are not
//       read: the fill loop only writes n_kv columns);
//   (c) at least one column is visible (the query's own cell must be).
// (c) is the one content assertion that needs no cell map: the token under
// test was just written into some cell of its own sequence at its own
// position, so exactly that cell must be visible.
bool case_multi_seq(llama_model * model, int vocab) {
    const int steps = 8;
    // valid in the unset and on arms (both run the binder); the off arm
    // reports it as skipped
    const bool skipped = g_arm == binder_arm::off;
    case_begin("multi_seq", 2, true, steps, "binder", skipped);
    if (skipped) {
        // not this arm's case: report the skip and touch nothing
        case_end("multi_seq", true, 0, true);
        return true;
    }

    shape_capture cap;
    llama_context * ctx = make_ctx(model, n_ctx_single, n_ctx_single, 2, true, &cap);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: context (multi_seq case)\n");
        case_end("multi_seq", false, 0, skipped);
        return false;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        std::fprintf(stderr, "FAIL: no memory (multi_seq case)\n");
        llama_free(ctx);
        case_end("multi_seq", false, 0, skipped);
        return false;
    }

    bool ok = true;
    const int sched[steps][2] = {
        { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 },
        { 0, 2 }, { 1, 2 }, { 0, 3 }, { 1, 3 },
    };

    for (int step = 0; ok && step < steps; ++step) {
        const int seq = sched[step][0];
        const int pos = sched[step][1];
        cap.reset();
        const int rc = decode_one(ctx, token_at(pos + seq*13, vocab), pos, seq);
        if (rc != 0) {
            std::fprintf(stderr, "FAIL: decode step=%d seq=%d rc=%d\n", step, seq, rc);
            ok = false;
            break;
        }
        if (!cap.seen_mask || !cap.seen_k || !cap.seen_v) {
            if (g_omit_root >= 0) {
                note_missing_observation(step, "multi_seq");
            }
            std::fprintf(stderr, "FAIL: missing tensor at step=%d\n", step);
            ok = false;
            break;
        }
        if (cap.mask_ne3 != 1) {
            std::fprintf(stderr, "FAIL: unified mask ne3=%lld want=1 at step=%d\n",
                         (long long) cap.mask_ne3, step);
            ok = false;
            break;
        }

        // The executed width is a POSITION span: the largest position of the
        // unified stream + 1, clamped to the KV capacity. Both sequences share
        // one stream, so two tokens at one position occupy one column and the
        // width must not grow past the positions it covers. used_pos_span_p1()
        // makes the public position maximum exact, so it is asserted here.
        int64_t span0 = llama_memory_seq_pos_max(mem, 0) + 1;
        int64_t span1 = llama_memory_seq_pos_max(mem, 1) + 1;
        if (span0 < 1) span0 = 1;
        if (span1 < 1) span1 = 1;
        int64_t width = span0 > span1 ? span0 : span1;
        if (width > (int64_t) llama_n_ctx_kv(ctx)) {
            width = (int64_t) llama_n_ctx_kv(ctx);
        }
        std::fprintf(stderr,
                "NKV_MULTI step=%d seq=%d pos=%d used0=%lld used1=%lld width=%lld "
                "mask_ne0=%lld k_ne2=%lld v_nkv=%lld pad_tail=%s\n",
                step, seq, pos, (long long) span0, (long long) span1, (long long) width,
                (long long) cap.mask_ne0, (long long) cap.k_nkv, (long long) cap.v_nkv,
                pad_tail_text(cap));
        if (cap.mask_ne0 != width || cap.k_nkv != width || cap.v_nkv != width) {
            std::fprintf(stderr,
                    "FAIL: shared width at step=%d want=%lld mask=%lld k=%lld v=%lld\n",
                    step, (long long) width, (long long) cap.mask_ne0,
                    (long long) cap.k_nkv, (long long) cap.v_nkv);
            ok = false;
            break;
        }
        if (!pad_tail_lane_ok(cap, width, "multi_seq", step)) {
            ok = false;
            break;
        }
        if (!nkv_runtime_ok(ctx, width, step)) {
            ok = false;
            break;
        }
        if (llama_memory_seq_pos_max(mem, seq) != pos) {
            std::fprintf(stderr, "FAIL: seq %d pos_max=%d want=%d\n",
                         seq, llama_memory_seq_pos_max(mem, seq), pos);
            ok = false;
            break;
        }

        // (b) + (c) on the single unified mask row
        if (!mask_row_props_ok(cap, pos, (int) width, 0)) {
            ok = false;
            break;
        }
        if (!binder_ok(ctx, step)) {
            ok = false;
            break;
        }
    }

    // Remove positions 0 and 1 of seq 0 and the whole of seq 1. The surviving
    // cells are seq 0's positions 2 and 3; the decode below writes position 4
    // and find_slot may reuse a freed low cell. The CELL span is therefore not
    // derivable from seq_pos_max. The width is a POSITION span, so the public
    // position maximum is exact here too and the width is asserted.
    if (ok) {
        if (!llama_memory_seq_rm(mem, 0, 0, 2) || !llama_memory_seq_rm(mem, 1, -1, -1)) {
            std::fprintf(stderr, "FAIL: seq_rm refused\n");
            ok = false;
        }
    }
    if (ok) {
        const int pos = 4;
        cap.reset();
        const int rc = decode_one(ctx, token_at(pos, vocab), pos, 0);
        if (rc != 0) {
            std::fprintf(stderr, "FAIL: decode after seq_rm rc=%d\n", rc);
            ok = false;
        } else if (!cap.seen_mask || !cap.seen_k || !cap.seen_v) {
            if (g_omit_root >= 0) {
                note_missing_observation(0, "multi_seq seq_rm");
            }
            std::fprintf(stderr, "FAIL: missing tensor after seq_rm\n");
            ok = false;
        } else {
            int64_t width = llama_memory_seq_pos_max(mem, 0) + 1;
            if (width < 1) width = 1;
            const int64_t capacity = (int64_t) llama_n_ctx_kv(ctx);
            if (width > capacity) {
                width = capacity;
            }
            std::fprintf(stderr,
                    "NKV_MULTI step=%d seq=0 pos=%d used0=%lld used1=0 width=%lld "
                    "mask_ne0=%lld k_ne2=%lld v_nkv=%lld pad_tail=%s\n",
                    steps, pos, (long long) width, (long long) width,
                    (long long) cap.mask_ne0, (long long) cap.k_nkv,
                    (long long) cap.v_nkv, pad_tail_text(cap));
            if (cap.mask_ne0 != width || cap.k_nkv != width || cap.v_nkv != width) {
                std::fprintf(stderr,
                        "FAIL: shared width after seq_rm want=%lld mask=%lld k=%lld v=%lld\n",
                        (long long) width, (long long) cap.mask_ne0,
                        (long long) cap.k_nkv, (long long) cap.v_nkv);
                ok = false;
            } else {
                ok = pad_tail_lane_ok(cap, width, "multi_seq seq_rm", steps) &&
                     nkv_runtime_ok(ctx, width, steps) &&
                     mask_row_props_ok(cap, pos, (int) width, 0) && binder_ok(ctx, steps);
            }
        }
    }

    llama_free(ctx);
    case_end("multi_seq", ok, steps + 1, skipped);
    return ok;
}


// Case: binder unset or on, non-unified, two streams. The binder must refuse
// and the run must continue on the stock/exact path.
//
// Non-unified means one KV stream per sequence (llama-kv-cache.cpp:202
// n_stream = unified ? 1 : n_seq_max). llama-context.cpp:4865 refuses any bind
// when kv_ctx->get_n_stream() > 1, so no patch may be applied and the
// counters must show patch_binds == 0. The diagnostics slot's PRESENCE is not
// the witness (the product allocates it lazily even when the gate is unset);
// its CONTENT is: the unset arm reports enabled=0 and env_disabled=0, the on
// arm reports envelope_disabled=1 and rebuild_fallbacks>=1. The executed width
// is still the true used span: get_n_kv() rounds by n_pad (1 for a plain
// cache) and, when the exact lane is armed on the on arm, get_n_kv() asks
// get_n_kv_exact() (llama-kv-cache.cpp:2095).
//
// Mask layout (llama-graph.cpp:87 and llama-kv-cache.cpp:2643): the mask is
// [n_kv, n_tokens/n_stream, 1, n_stream] with
// n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq. Every decode here
// carries ONE token, so ubatch.n_seqs_unq == 1 and the mask is
// [n_kv, 1, 1, 1] even though the cache has n_seq_max == 2. The fill loop
// indexes a row as i = s*n_tps + ii, which is 0, so the current token's row
// is row 0. A per-stream row model would index outside this tensor.
//
// The capacity is arm-dependent. On the unset arm cpu_default requires
// kv_unified || n_seq_max == 1 (llama-context.cpp:556), so the multi-stream
// cache is not qualified: the capacity sites pad n_ctx to
// GGML_PAD(300, 256) = 512, then n_ctx_seq = n_ctx_kv = 256. On the on arm
// the explicit gate keeps the requested capacity: ceil(300/2) = 150.
bool case_multi_stream(llama_model * model, int vocab) {
    const int steps = 4;
    // valid in the unset and on arms (both run the binder); the off arm
    // reports it as skipped
    const bool skipped = g_arm == binder_arm::off;
    case_begin("multi_stream", 2, false, steps, "binder", skipped);
    if (skipped) {
        // not this arm's case: report the skip and touch nothing
        case_end("multi_stream", true, 0, true);
        return true;
    }

    shape_capture cap;
    llama_context * ctx = make_ctx(model, n_ctx_single, 0, 2, false, &cap);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: context (multi_stream case)\n");
        case_end("multi_stream", false, 0, skipped);
        return false;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        std::fprintf(stderr, "FAIL: no memory (multi_stream case)\n");
        llama_free(ctx);
        case_end("multi_stream", false, 0, skipped);
        return false;
    }

    const uint32_t stock_width = llama_n_ctx_kv(ctx);
    const uint32_t want_cap = g_arm == binder_arm::on ? 150u : 256u;
    std::fprintf(stderr,
            "NKV_MULTI_STREAM n_ctx=%u n_ctx_seq=%u n_ctx_kv=%u n_seq_max=%u stock_width=%u\n",
            llama_n_ctx(ctx), llama_n_ctx_seq(ctx), llama_n_ctx_kv(ctx),
            llama_n_seq_max(ctx), stock_width);
    if (stock_width != want_cap) {
        std::fprintf(stderr,
                "FAIL: stock width=%u want=%u for arm %s "
                "(unset: GGML_PAD(300/2, 256); on: ceil(300/2))\n",
                stock_width, want_cap, g_arm == binder_arm::on ? "on" : "unset");
        llama_free(ctx);
        case_end("multi_stream", false, 0, skipped);
        return false;
    }

    bool ok = true;
    const int sched[steps][2] = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };

    for (int step = 0; ok && step < steps; ++step) {
        const int seq = sched[step][0];
        const int pos = sched[step][1];
        cap.reset();
        const int rc = decode_one(ctx, token_at(pos + seq*13, vocab), pos, seq);
        if (rc != 0) {
            std::fprintf(stderr, "FAIL: decode step=%d seq=%d rc=%d\n", step, seq, rc);
            ok = false;
            break;
        }
        if (!cap.seen_mask || !cap.seen_k || !cap.seen_v) {
            if (g_omit_root >= 0) {
                note_missing_observation(step, "multi_stream");
            }
            std::fprintf(stderr, "FAIL: missing tensor at step=%d\n", step);
            ok = false;
            break;
        }
        // non-unified, one token per decode: the mask carries one stream and
        // one token row, [n_kv, 1, 1, 1] (llama-graph.cpp:87)
        if (cap.mask_ne3 != 1 || cap.mask_ne1 != 1) {
            std::fprintf(stderr, "FAIL: mask ne1=%lld ne3=%lld want=1/1 at step=%d "
                         "(one token per decode => n_seqs_unq=1)\n",
                         (long long) cap.mask_ne1, (long long) cap.mask_ne3, step);
            ok = false;
            break;
        }
        // no node carries the pad tail on either arm; the field is printed as
        // none and is not asserted in this case
        std::fprintf(stderr,
                "NKV_MULTI_STREAM step=%d seq=%d pos=%d mask_ne0=%lld mask_ne1=%lld "
                "mask_ne3=%lld k_ne2=%lld v_nkv=%lld pad_tail=%s\n",
                step, seq, pos, (long long) cap.mask_ne0, (long long) cap.mask_ne1,
                (long long) cap.mask_ne3, (long long) cap.k_nkv, (long long) cap.v_nkv,
                pad_tail_text(cap));
        // the refusal: no bind is applied, so the executed width is the true
        // used span, not the capacity. get_n_kv() rounds by n_pad (1 for a
        // plain cache); the schedule is 1,1,2,2.
        const int64_t want_width = step < 2 ? 1 : 2;
        if (cap.mask_ne0 != want_width ||
                cap.k_nkv != want_width || cap.v_nkv != want_width) {
            std::fprintf(stderr,
                    "FAIL: multi-stream width at step=%d want %lld mask=%lld k=%lld v=%lld\n",
                    step, (long long) want_width, (long long) cap.mask_ne0,
                    (long long) cap.k_nkv, (long long) cap.v_nkv);
            ok = false;
            break;
        }

        // the mask of this decode's token, by property (no cell map). The
        // only row is row 0 (the token's own row).
        if (!mask_row_props_ok(cap, pos, (int) want_width, 0)) {
            ok = false;
            break;
        }

        // The bind must be refused on both arms. The diagnostics slot's
        // PRESENCE is not the witness: the product allocates it lazily even
        // when the gate is unset. Its CONTENT is:
        //   unset: enabled=0 and envelope_disabled=0, no bind;
        //   on:    envelope_disabled=1 and rebuild_fallbacks>=1, no bind.
        llama_union_diagnostics d = {};
        const bool has_state = llama_get_union_diagnostics(ctx, &d);
        std::fprintf(stderr,
                "NKV_MULTI_STREAM_BIND step=%d state=%d enabled=%d env_disabled=%d "
                "patch_binds=%llu fallbacks=%llu\n",
                step, int(has_state), int(d.enabled), int(d.envelope_disabled),
                (unsigned long long) d.patch_binds,
                (unsigned long long) d.rebuild_fallbacks);
        if (!has_state) {
            std::fprintf(stderr,
                    "FAIL: no binder diagnostics at step=%d "
                    "(the slot's content is the witness)\n", step);
            ok = false;
            break;
        }
        if (d.patch_binds != 0) {
            std::fprintf(stderr, "FAIL: multi-stream bind was not refused at step=%d\n", step);
            ok = false;
            break;
        }
        if (g_arm == binder_arm::unset) {
            if (d.enabled != 0 || d.envelope_disabled != 0) {
                std::fprintf(stderr,
                        "FAIL: unset arm diagnostics at step=%d enabled=%d "
                        "envelope_disabled=%d (want 0/0)\n",
                        step, int(d.enabled), int(d.envelope_disabled));
                ok = false;
                break;
            }
        } else if (d.envelope_disabled != 1 || d.rebuild_fallbacks < 1) {
            // A bind that is never ATTEMPTED would also leave patch_binds==0.
            // The refusal must show its own evidence: the fresh-build bind is
            // refused (llama-context.cpp:4865), which disables the envelope
            // and counts one rebuild fallback (llama-context.cpp:6219-6220).
            std::fprintf(stderr,
                    "FAIL: multi-stream refusal evidence at step=%d "
                    "envelope_disabled=%d rebuild_fallbacks=%llu "
                    "(want envelope_disabled=1 and rebuild_fallbacks>=1)\n",
                    step, int(d.envelope_disabled),
                    (unsigned long long) d.rebuild_fallbacks);
            ok = false;
            break;
        }
    }

    llama_free(ctx);
    case_end("multi_stream", ok, steps, skipped);
    return ok;
}



// Case: binder unset, unified, capacity 257. One sequence at positions
// 0..256: the executed width crosses 1, 255, 256 and 257, and the last step
// runs at the full physical capacity with no headroom above it.
//
// A width that moves DOWN cannot be produced through the public API here:
// llama_memory_seq_rm frees the cells it removes, and the stream head then
// reuses the lowest free cell, so the used span stays where it was. The
// `result > min_size` guard of get_n_kv is likewise unreachable (see the file
// header); neither is faked here.
bool case_widths(llama_model * model, int vocab) {
    const int steps = n_ctx_widths;
    // valid in the unset and on arms (both run the binder); the off arm
    // reports it as skipped
    const bool skipped = g_arm == binder_arm::off;
    case_begin("widths", 1, true, steps, "binder", skipped);
    if (skipped) {
        // not this arm's case: report the skip and touch nothing
        case_end("widths", true, 0, true);
        return true;
    }

    shape_capture cap;
    llama_context * ctx = make_ctx(model, n_ctx_widths, n_ctx_widths, 1, true, &cap);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: context (widths case)\n");
        case_end("widths", false, 0, skipped);
        return false;
    }
    if (llama_n_ctx_kv(ctx) != (uint32_t) n_ctx_widths) {
        std::fprintf(stderr, "FAIL: n_ctx_kv=%u want=%d\n", llama_n_ctx_kv(ctx), n_ctx_widths);
        llama_free(ctx);
        case_end("widths", false, 0, skipped);
        return false;
    }

    bool ok = true;
    int pos = 0;
    for (int step = 0; ok && step < steps; ++step) {
        cap.reset();
        const int rc = decode_one(ctx, token_at(pos, vocab), pos, 0);
        if (rc != 0) {
            std::fprintf(stderr, "FAIL: decode step=%d rc=%d\n", step, rc);
            ok = false;
            break;
        }
        pos += 1;

        if (!cap.seen_mask || !cap.seen_k || !cap.seen_v) {
            if (g_omit_root >= 0) {
                note_missing_observation(step, "widths");
            }
            std::fprintf(stderr, "FAIL: missing tensor at step=%d\n", step);
            ok = false;
            break;
        }
        std::fprintf(stderr,
                "NKV_WIDTH step=%d pos=%d mask_ne0=%lld k_ne2=%lld v_nkv=%lld "
                "at_capacity=%d pad_tail=%s\n",
                step, pos, (long long) cap.mask_ne0, (long long) cap.k_nkv,
                (long long) cap.v_nkv, int(pos == n_ctx_widths), pad_tail_text(cap));
        if (cap.mask_ne0 != pos || cap.k_nkv != pos || cap.v_nkv != pos) {
            std::fprintf(stderr,
                    "FAIL: padded width at step=%d pos=%d mask=%lld k=%lld v=%lld\n",
                    step, pos, (long long) cap.mask_ne0,
                    (long long) cap.k_nkv, (long long) cap.v_nkv);
            ok = false;
            break;
        }
        if (!pad_tail_lane_ok(cap, pos, "widths", step)) {
            ok = false;
            break;
        }
        if (!nkv_runtime_ok(ctx, pos, step)) {
            ok = false;
            break;
        }
        if (!binder_ok(ctx, step)) {
            ok = false;
            break;
        }
    }

    llama_free(ctx);
    case_end("widths", ok, steps, skipped);
    return ok;
}


// Case: binder explicitly off. The stock control.
//
// The gate is not set here. This case runs only when the runner started the
// process with GGML_TP5_UNION_BINDER=0 (main() skips it in every other arm),
// and the context reads the gate once at construction (llama-context.cpp:508).
// Setting it here would hide a runner that started the wrong arm.
bool case_off(llama_model * model, int vocab) {
    const int steps = steps_single;
    // valid only in the off arm; any other arm reports it as skipped
    const bool skipped = g_arm != binder_arm::off;
    case_begin("off", 1, true, steps, "off", skipped);
    if (skipped) {
        // not this arm's case: report the skip and touch nothing
        case_end("off", true, 0, true);
        return true;
    }

    shape_capture cap;
    llama_context * ctx = make_ctx(model, n_ctx_single, n_ctx_single, 1, true, &cap);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: context (off case)\n");
        case_end("off", false, 0, skipped);
        return false;
    }

    // The gate-off capacity pad is asserted, not assumed: n_ctx=300 pads to
    // GGML_PAD(300, 256) = 512 (llama-context.cpp:669), and a unified cache
    // copies n_ctx to n_ctx_seq and then to n_ctx_kv
    // (llama-context.cpp:710-711, :735).
    const int64_t capacity = ((n_ctx_single + 255)/256)*256;
    if (llama_n_ctx_kv(ctx) != (uint32_t) capacity) {
        std::fprintf(stderr, "FAIL: off capacity n_ctx_kv=%u want=%lld\n",
                     llama_n_ctx_kv(ctx), (long long) capacity);
        llama_free(ctx);
        case_end("off", false, 0, skipped);
        return false;
    }

    bool ok = true;
    // the stock control must not allocate binder state
    llama_union_diagnostics d = {};
    if (llama_get_union_diagnostics(ctx, &d)) {
        std::fprintf(stderr, "FAIL: off arm allocated binder state\n");
        ok = false;
    }

    const uint64_t builds_init = llama_graph_build_count(ctx);
    int pos = 0;
    int prev_pos = -1;          // the position of the previous observed step
    int64_t prev_width = -1;    // the width observed at that step
    for (int step = 0; ok && step < steps; ++step) {
        cap.reset();
        const int rc = decode_one(ctx, token_at(pos, vocab), pos, 0);
        if (rc != 0) {
            std::fprintf(stderr, "FAIL: off decode step=%d rc=%d\n", step, rc);
            ok = false;
            break;
        }
        pos += 1;
        if (step == 0 || step == 255 || step == 256 || step == steps - 1) {
            if (!cap.seen_mask || !cap.seen_k || !cap.seen_v) {
                if (g_omit_root >= 0) {
                    note_missing_observation(step, "off");
                }
                std::fprintf(stderr, "FAIL: off missing tensor at step=%d\n", step);
                ok = false;
                break;
            }
            // no binder here, so no node carries the pad tail: the field is
            // printed as none and is not asserted in this arm
            std::fprintf(stderr,
                    "OFF_ARM step=%d pos=%d mask_ne0=%lld k_ne2=%lld v_nkv=%lld "
                    "pad_tail=%s\n",
                    step, pos, (long long) cap.mask_ne0, (long long) cap.k_nkv,
                    (long long) cap.v_nkv, pad_tail_text(cap));
            // Gate off: llama-context.cpp pads n_ctx to GGML_PAD(n_ctx, 256)
            // and n_ctx_seq/n_ctx_kv follow it, so the CAPACITY is 512 here
            // (asserted above, before the loop).
            // The executed width is NOT that bucket: the 256-cell floor in
            // llama_kv_cache::get_n_kv() was deleted (8de2c4a), and the
            // surviving rounding uses n_pad, which is 1 for a plain cache
            // (llama-kv-cache.h n_pad, llama-model.cpp passes the literal 1).
            // The stock path therefore reports the TRUE USED SPAN: pos + 1
            // after this step's token, clamped to the capacity.
            //
            // This is what keeps the `off` arm a genuine control. It reaches
            // the same width the exact arm binds, but by a different path: no
            // binder state, no bind, and a graph that rebuilds whenever the
            // span moves (see OFF_ARM builds below), where the exact arm
            // binds in place: one build and zero rebuild fallbacks.
            if (cap.mask_ne0 != cap.k_nkv || cap.mask_ne0 != cap.v_nkv) {
                std::fprintf(stderr,
                        "FAIL: off mask/K/V widths disagree at step=%d mask=%lld k=%lld v=%lld\n",
                        step, (long long) cap.mask_ne0,
                        (long long) cap.k_nkv, (long long) cap.v_nkv);
                ok = false;
                break;
            }
            const int64_t want     = pos < capacity ? pos : n_ctx_single;
            if (cap.mask_ne0 != want) {
                std::fprintf(stderr,
                        "FAIL: off width at step=%d pos=%d want=%lld mask=%lld\n",
                        step, pos, (long long) want, (long long) cap.mask_ne0);
                ok = false;
                break;
            }
            // The width must FOLLOW the used span, not sit on a bucket.
            //
            // This runs at every observed step, including the ones past 256.
            // It used to be suppressed for pos <= 255 because the 256-cell
            // floor made the width sit at 256 until the span passed it; with
            // the floor gone (n_pad == 1) the width follows the span at every
            // step, so suppressing it there would skip exactly the boundary
            // this arm exists to cross.
            //
            // The one region where the width cannot follow is above the
            // CAPACITY clamp in `want`: past `capacity` the width stops at
            // n_ctx_single. That is a clamp, not a bucket, so the check is
            // bounded by `capacity` and stated as such. In this arm the clamp
            // is unreachable by construction (steps_single == n_ctx_single ==
            // 300 < capacity == 512), so every observed step is checked; the
            // bound stays so the check cannot false-fail if the schedule ever
            // grows past the capacity.
            //
            // The comparison uses the PREVIOUS observation's width, not the
            // first one: the width must grow by the span delta between the
            // two observations. The old first-width form mixed a width from
            // pos 1 with the delta from pos 256 to 257 and demanded 2.
            if (prev_width >= 0 && pos <= capacity &&
                    cap.mask_ne0 != prev_width + (pos - prev_pos)) {
                std::fprintf(stderr,
                        "FAIL: off width did not follow the used span at step=%d "
                        "(%lld -> %lld, pos %d -> %d)\n",
                        step, (long long) prev_width, (long long) cap.mask_ne0,
                        prev_pos, pos);
                ok = false;
                break;
            }
            prev_pos   = pos;
            prev_width = cap.mask_ne0;
        }
    }

    // The stock control must build at least once. It is NOT the reuse arm:
    // the graph's n_kv/mask width follows the used span, so the stock path
    // rebuilds whenever the span moves (observed builds == steps on this
    // growing schedule). Reuse without a per-step rebuild is the BINDER
    // arm's claim (case_single asserts exactly one build); the off arm is
    // the control that proves the stock path is a different path.
    const uint64_t off_builds = llama_graph_build_count(ctx) - builds_init;
    std::fprintf(stderr, "OFF_ARM builds=%llu steps=%d\n",
                 (unsigned long long) off_builds, steps);
    if (ok && off_builds < 1) {
        std::fprintf(stderr, "FAIL: off graph builds=%llu want >= 1\n",
                     (unsigned long long) off_builds);
        ok = false;
    }
    if (ok) {
        std::fprintf(stderr, "OFF_ARM_BUILDS ok=1 builds=%llu steps=%d\n",
                     (unsigned long long) off_builds, steps);
    }

    llama_free(ctx);
    case_end("off", ok, steps, skipped);
    return ok;
}

} // namespace

// The arm as the runner set it. Read once; never written.
struct arm_env {
    const char * binder;
    const char * ntok;
    const char * reuse_disable;
};

const char * env_value(const char * value) {
    return value == nullptr ? "unset" : value;
}

arm_env read_arm_env() {
    return arm_env{
        /*.binder        =*/ env_value(std::getenv("GGML_TP5_UNION_BINDER")),
        /*.ntok          =*/ env_value(std::getenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK")),
        /*.reuse_disable =*/ env_value(std::getenv("LLAMA_GRAPH_REUSE_DISABLE")),
    };
}

const char * binder_arm_name(binder_arm arm) {
    switch (arm) {
        case binder_arm::unset: return "unset";
        case binder_arm::off:   return "off";
        case binder_arm::on:    return "on";
    }
    return "?";
}

bool parse_binder_arm(const char * value, binder_arm & out) {
    if (value == nullptr || value[0] == '\0' || strcmp(value, "unset") == 0) {
        out = binder_arm::unset;
        return true;
    }
    if (strcmp(value, "0") == 0) {
        out = binder_arm::off;
        return true;
    }
    if (strcmp(value, "1") == 0) {
        out = binder_arm::on;
        return true;
    }
    return false;
}

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 4) {
        std::fprintf(stderr,
                     "usage: %s model.gguf [--negative omit-mask|omit-k|omit-v|omit-pad-tail]\n",
                     argv[0]);
        return 2;
    }
    if (argc == 4) {
        if (strcmp(argv[2], "--negative") != 0) {
            std::fprintf(stderr, "FAIL: unknown option %s\n", argv[2]);
            return 2;
        }
        if      (strcmp(argv[3], "omit-mask")     == 0) g_omit_root = int(nkv_root::mask);
        else if (strcmp(argv[3], "omit-k")        == 0) g_omit_root = int(nkv_root::k);
        else if (strcmp(argv[3], "omit-v")        == 0) g_omit_root = int(nkv_root::v);
        else if (strcmp(argv[3], "omit-pad-tail") == 0) g_omit_root = int(nkv_root::pad_tail);
        else {
            std::fprintf(stderr,
                         "FAIL: unknown --negative target %s (want omit-mask, omit-k, "
                         "omit-v or omit-pad-tail)\n",
                         argv[3]);
            return 2;
        }
    }

    // The arm belongs to the runner. Reading it here changes nothing about
    // the process environment; it only names the path this run must take.
    const arm_env arm = read_arm_env();
    if (!parse_binder_arm(arm.binder, g_arm)) {
        std::fprintf(stderr,
                     "FAIL: GGML_TP5_UNION_BINDER=%s is not an arm this test runs "
                     "(want unset, 0 or 1)\n", arm.binder);
        return 2;
    }
    std::fprintf(stderr,
                 "NKV_ARM binder=%s ntok=%s reuse_disable=%s negative=%s\n",
                 arm.binder, arm.ntok, arm.reuse_disable,
                 g_omit_root < 0 ? "none" : nkv_root_name((nkv_root) g_omit_root));

    llama_backend_init();

    auto mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = { nullptr };
    mp.devices      = devices;
    mp.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        std::fprintf(stderr, "FAIL: model load\n");
        llama_backend_free();
        return 1;
    }

    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (vocab <= 3) {
        std::fprintf(stderr, "FAIL: vocab=%d\n", vocab);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Every case is called in every arm. A case whose arm this is not reports
    // itself as skipped=1, contributes nothing to the totals, and is not
    // consulted for the exit status: a case that did not run cannot fail, and
    // must not be able to pass either.
    struct case_run {
        const char * name;
        binder_arm   arm;
        bool         ok;
    };
    bool ok = true;
    const case_run runs[] = {
        { "single",       binder_arm::unset, case_single       (model, vocab) },
        { "multi_seq",    binder_arm::unset, case_multi_seq    (model, vocab) },
        { "multi_stream", binder_arm::unset, case_multi_stream (model, vocab) },
        { "widths",       binder_arm::unset, case_widths       (model, vocab) },
        { "off",          binder_arm::off,   case_off          (model, vocab) },
    };
    for (const case_run & c : runs) {
        if (g_arm != c.arm) {
            continue;
        }
        if (!c.ok) {
            std::fprintf(stderr, "FAIL: case %s failed in arm %s\n",
                         c.name, binder_arm_name(g_arm));
            ok = false;
        }
    }

    llama_model_free(model);
    llama_backend_free();

    const bool all_ok = ok && g_all_ok;
    // A run that skipped every case compared nothing. The footer must not
    // report ok=1 for it: cases and steps are the work the run owes, and the
    // gate pins them. Keep `all_ok` for the negative arm below. A zero-work
    // run is UNPROVEN, not FAIL: nothing was compared, so the run cannot say
    // the observation is wrong. Every other path keeps the old verdict.
    const bool footer_ok = all_ok && g_cases_run != 0 && g_steps_run != 0;
    const bool footer_unproven = all_ok && !footer_ok;
    if (footer_unproven) {
        std::fprintf(stderr, "UNPROVEN: the run compared no case (cases=%d steps=%d)\n",
                     g_cases_run, g_steps_run);
        std::fprintf(stderr, "NKV_EXACT_END ok=2 reason=unproven:no_case_ran cases=%d steps=%d\n",
                     g_cases_run, g_steps_run);
    } else {
        std::fprintf(stderr, "NKV_EXACT_END ok=%d cases=%d steps=%d\n",
                     int(footer_ok), g_cases_run, g_steps_run);
    }

    if (g_omit_root >= 0) {
        // A negative run succeeds only when the omission was DETECTED. Three
        // things must all hold, and none of them is "the process exited
        // nonzero":
        //
        //   1. the run did not pass (all_ok == 0). A pass means the gate
        //      cannot tell a missing observation from a present one.
        //   2. a case actually recorded the missing observation for the
        //      omitted root (g_negative.missing_seen, or pad_tail_seen for
        //      omit-pad-tail). A run that failed at model load, at context
        //      creation, or on the wrong arm also exits nonzero, and proves
        //      nothing about the observation.
        //   3. the evidence names the root this run was told to omit.
        //
        // The named line is printed by note_missing_observation() or
        // note_missing_pad_tail() at the same moment the evidence is
        // recorded, so a log carrying NKV_NEGATIVE ok=1 always carries that
        // line for the same root. The pad tail has its own evidence record
        // because it is not one of the three n_kv roots: it is a width on a
        // node, and it can be missing while the mask is present.
        const nkv_root want = (nkv_root) g_omit_root;
        const bool pad_tail = want == nkv_root::pad_tail;
        const bool seen     = pad_tail ? g_negative.pad_tail_seen : g_negative.missing_seen;
        if (all_ok) {
            std::fprintf(stderr,
                         "FAIL: negative run omitting %s passed; the gate cannot "
                         "see a missing observation\n",
                         nkv_root_name(want));
            return 1;
        }
        if (!seen) {
            std::fprintf(stderr,
                         "FAIL: negative run omitting %s failed without the named "
                         "'missing %s' line; the failure came from somewhere "
                         "other than the omitted observation\n",
                         nkv_root_name(want),
                         pad_tail ? "pad tail" : "tensor");
            return 1;
        }
        if (!pad_tail && g_negative.missing_root != want) {
            std::fprintf(stderr,
                         "FAIL: negative run omitting %s recorded a missing "
                         "observation for %s\n",
                         nkv_root_name(want),
                         nkv_root_name(g_negative.missing_root));
            return 1;
        }
        std::fprintf(stderr, "NKV_NEGATIVE ok=1 omitted=%s\n",
                     nkv_root_name(want));
        return 0;
    }

    return footer_ok ? 0 : (footer_unproven ? 2 : 1);
}
