// Ownership test for llama_kv_cells::journal_.
//
// The journal borrow is private, so every check below is behavioural. A bound
// object is the live writer of its stream and is never assigned; both
// assignment operators require an unbound destination. The replay checks use
// the real restore_from_journal path and assert the user-visible cell state.
#include "llama-kv-cells.h"

#include <cstdio>
#include <cstdlib>
#include <new>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#include <cerrno>
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

static int failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

// One-shot allocation failure for the strong-exception-guarantee test. The
// next operator new throws instead of returning storage.
static bool fail_next_alloc = false;

void * operator new(std::size_t n) {
    if (fail_next_alloc) {
        fail_next_alloc = false;
        throw std::bad_alloc();
    }
    if (void * p = std::malloc(n ? n : 1)) {
        return p;
    }
    throw std::bad_alloc();
}

void * operator new[](std::size_t n) {
    return operator new(n);
}

void operator delete(void * p) noexcept {
    std::free(p);
}

void operator delete[](void * p) noexcept {
    std::free(p);
}

void operator delete(void * p, std::size_t) noexcept {
    std::free(p);
}

void operator delete[](void * p, std::size_t) noexcept {
    std::free(p);
}

static void seed(llama_kv_cells & cells) {
    cells.resize(4);
}

static void add_cell(llama_kv_cells & cells, uint32_t i, llama_pos p, llama_seq_id s) {
    cells.pos_set(i, p);
    cells.seq_add(i, s);
}

// Copy construction: the snapshot must not inherit the borrow.
static void test_copy_ctor_drops_journal() {
    llama_kv_cells src;
    seed(src);
    cell_journal j;
    j.begin(4);
    src.set_journal(&j);
    src.pos_set(0, 10);
    const size_t n0 = j.size();

    llama_kv_cells snap(src);
    snap.pos_set(1, 20); // must not record
    CHECK(j.size() == n0);

    src.pos_set(2, 30); // the source keeps its borrow
    CHECK(j.size() == n0 + 1);
}

// Copy assignment into an unbound destination: the destination stays unbound
// and never inherits the source's journal.
static void test_copy_assign_does_not_inherit_journal() {
    llama_kv_cells src;
    seed(src);
    cell_journal j;
    j.begin(4);
    src.set_journal(&j);
    src.pos_set(0, 10);
    const size_t n0 = j.size();

    llama_kv_cells snap;
    seed(snap);
    snap = src;
    snap.pos_set(1, 20);
    CHECK(j.size() == n0);

    src.pos_set(2, 30);
    CHECK(j.size() == n0 + 1);
}

// Move construction: the destination continues the trial, the source is
// cleared and must not write into the journal afterwards.
static void test_move_ctor_transfers_journal() {
    llama_kv_cells src;
    seed(src);
    cell_journal j;
    j.begin(4);
    src.set_journal(&j);
    src.pos_set(0, 10);
    const size_t n0 = j.size();

    llama_kv_cells moved(std::move(src));
    moved.pos_set(1, 20);
    CHECK(j.size() == n0 + 1);

    seed(src); // a moved-from object is valid but unspecified; make it usable
    src.pos_set(2, 30);
    CHECK(j.size() == n0 + 1);
}

// Move assignment into an unbound destination: it adopts the source's borrow,
// like move construction.
static void test_move_assign_unbound_adopts_journal() {
    llama_kv_cells src;
    seed(src);
    cell_journal j;
    j.begin(4);
    src.set_journal(&j);
    src.pos_set(0, 10);
    const size_t n0 = j.size();

    llama_kv_cells moved;
    seed(moved);
    moved = std::move(src);
    moved.pos_set(1, 20);
    CHECK(j.size() == n0 + 1);

    seed(src);
    src.pos_set(2, 30);
    CHECK(j.size() == n0 + 1);
}

// Copy assignment into an unbound destination, then replay the source's
// journal: the source must return to its pre-trial cell state.
static void test_copy_assign_replay_source_state() {
    llama_kv_cells src;
    seed(src);
    add_cell(src, 0, 7, 2);
    add_cell(src, 1, 11, 2);

    cell_journal j;
    j.begin(4);
    src.set_journal(&j);

    src.rm(0);
    src.seq_rm(1, 2);
    CHECK(src.get_used() == 0);

    llama_kv_cells snap;
    seed(snap);
    snap = src; // unbound destination: no journal is inherited
    CHECK(snap.get_used() == 0);
    snap.pos_set(2, 30); // must not record into the source's journal
    CHECK(j.size() == 2);

    src.set_journal(nullptr);
    j.replay([&](cell_journal::entry & e) {
        CHECK(src.restore_from_journal(j, e));
    });

    CHECK(src.get_used() == 2);
    CHECK(src.pos_get(0) == 7);
    CHECK(src.seq_has(0, 2));
    CHECK(src.pos_get(1) == 11);
    CHECK(src.seq_has(1, 2));
    CHECK(src.seq_pos_min(2) == 7);
    CHECK(src.seq_pos_max(2) == 11);

    // the snapshot is independent of the source's replay
    CHECK(snap.get_used() == 1);
    CHECK(snap.pos_get(2) == 30);
}

// Move assignment into an unbound destination, then replay the transferred
// journal: the destination must return to the source's pre-move cell state.
static void test_move_assign_replay_destination_state() {
    llama_kv_cells src;
    seed(src);
    add_cell(src, 0, 7, 2);
    add_cell(src, 1, 11, 2);

    cell_journal j;
    j.begin(4);
    src.set_journal(&j);

    src.rm(0);
    src.seq_rm(1, 2);
    CHECK(src.get_used() == 0);

    llama_kv_cells dst;
    seed(dst);
    dst = std::move(src); // unbound destination: adopts the source's journal
    CHECK(dst.get_used() == 0);

    dst.pos_set(2, 30); // continues the trial in the destination
    CHECK(j.size() == 3);

    dst.set_journal(nullptr);
    j.replay([&](cell_journal::entry & e) {
        CHECK(dst.restore_from_journal(j, e));
    });

    CHECK(dst.get_used() == 2);
    CHECK(dst.pos_get(0) == 7);
    CHECK(dst.seq_has(0, 2));
    CHECK(dst.pos_get(1) == 11);
    CHECK(dst.seq_has(1, 2));
    CHECK(dst.seq_pos_min(2) == 7);
    CHECK(dst.seq_pos_max(2) == 11);
    CHECK(dst.get_used_set().count(2) == 0);
}

// A real trial: mutate used cells under the journal, replay it, and check the
// user-visible cell state, not only the entry count.
static void test_restore_real_cell_state() {
    llama_kv_cells cells;
    seed(cells);
    add_cell(cells, 0, 7, 2);
    add_cell(cells, 1, 11, 2);
    llama_kv_cell_ext ext;
    ext.x = 3;
    ext.y = 4;
    cells.ext_set(0, ext);

    cell_journal j;
    j.begin(4);
    cells.set_journal(&j);

    cells.rm(0);
    cells.seq_rm(1, 2);

    CHECK(cells.get_used() == 0);
    CHECK(cells.seq_pos_min(2) == -1);

    cells.set_journal(nullptr);
    j.replay([&](cell_journal::entry & e) {
        CHECK(cells.restore_from_journal(j, e));
    });

    CHECK(cells.get_used() == 2);
    CHECK(cells.pos_get(0) == 7);
    CHECK(cells.seq_has(0, 2));
    CHECK(cells.ext_get(0).x == 3);
    CHECK(cells.ext_get(0).y == 4);
    CHECK(cells.pos_get(1) == 11);
    CHECK(cells.seq_has(1, 2));
    CHECK(cells.seq_pos_min(2) == 7);
    CHECK(cells.seq_pos_max(2) == 11);
    CHECK(j.size() == 0);
}

// A failed copy assignment must leave the unbound destination's data
// untouched (copy-and-swap), and the source's trial must keep working.
static void test_copy_assign_oom_keeps_destination() {
    llama_kv_cells dst;
    seed(dst);
    dst.pos_set(0, 5);
    dst.seq_add(0, 1);
    dst.pos_set(1, 6);

    llama_kv_cells src;
    seed(src);
    src.pos_set(0, 9);
    src.seq_add(0, 3);
    src.pos_set(2, 10);

    cell_journal js;
    js.begin(4);
    src.set_journal(&js);
    src.pos_set(3, 12);
    const size_t js0 = js.size();

    bool threw = false;
    fail_next_alloc = true;
    try {
        dst = src;
    } catch (const std::bad_alloc &) {
        threw = true;
    }
    fail_next_alloc = false;

    CHECK(threw);
    CHECK(dst.get_used() == 2);
    CHECK(dst.pos_get(0) == 5);
    CHECK(dst.seq_has(0, 1));
    CHECK(dst.pos_get(1) == 6);
    CHECK(dst.get_used_set().count(2) == 0);
    CHECK(dst.seq_pos_min(1) == 5);

    // the destination is still an unbound snapshot and can be bound later
    cell_journal jd;
    jd.begin(4);
    dst.set_journal(&jd);
    dst.pos_set(2, 7);
    CHECK(jd.size() == 1);

    // the failed copy did not touch the source or its trial
    CHECK(src.get_used() == 3);
    CHECK(src.pos_get(0) == 9);
    CHECK(src.seq_has(0, 3));
    CHECK(src.pos_get(2) == 10);
    CHECK(js.size() == js0);

    src.set_journal(nullptr);
    js.replay([&](cell_journal::entry & e) {
        CHECK(src.restore_from_journal(js, e));
    });
    CHECK(src.get_used() == 2);
    CHECK(src.pos_get(0) == 9);
    CHECK(src.pos_get(2) == 10);
    CHECK(src.get_used_set().count(3) == 0);
}

// Assignment over a bound destination is a contract violation. GGML_ASSERT is
// active in every build, so the child must die with SIGABRT.
#if defined(__unix__) || defined(__APPLE__)
static bool wait_for_abort(pid_t pid) {
    int status = 0;
    pid_t got;
    do {
        got = waitpid(pid, &status, 0);
    } while (got < 0 && errno == EINTR);
    return got == pid && WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

static void expect_bound_assign_abort(bool move_assign) {
    const pid_t pid = fork();
    if (pid < 0) {
        CHECK(false);
        return;
    }
    if (pid == 0) {
        // ggml_abort prints a backtrace through gdb; the expected abort must
        // not launch it.
        setenv("GGML_NO_BACKTRACE", "1", 1);

        llama_kv_cells dst;
        seed(dst);
        cell_journal jd;
        jd.begin(4);
        dst.set_journal(&jd);
        dst.pos_set(0, 5);

        llama_kv_cells src;
        seed(src);
        src.pos_set(0, 9);

        if (move_assign) {
            dst = std::move(src);
        } else {
            dst = src;
        }

        _exit(0); // the invariant was not enforced
    }

    CHECK(wait_for_abort(pid));
}

static void test_bound_destination_assignment_aborts() {
    expect_bound_assign_abort(false);
    expect_bound_assign_abort(true);
}
#else
static void test_bound_destination_assignment_aborts() {
    // GGML_ASSERT aborts in every build; the death check is POSIX-only.
}
#endif

int main() {
    test_copy_ctor_drops_journal();
    test_copy_assign_does_not_inherit_journal();
    test_move_ctor_transfers_journal();
    test_move_assign_unbound_adopts_journal();
    test_copy_assign_replay_source_state();
    test_move_assign_replay_destination_state();
    test_restore_real_cell_state();
    test_copy_assign_oom_keeps_destination();
    test_bound_destination_assignment_aborts();

    if (failures != 0) {
        std::fprintf(stderr, "KV_CELLS_JOURNAL_OWNERSHIP_TEST FAIL %d\n", failures);
        return 1;
    }
    std::printf("KV_CELLS_JOURNAL_OWNERSHIP_TEST PASS\n");
    return 0;
}
