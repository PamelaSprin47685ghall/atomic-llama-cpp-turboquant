#pragma once

#include "llama.h"
#include "llama-cparams.h"
#include "llama-rerot.h"

#include <atomic>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

struct llama_kv_cell_ext {
    // 2D spatial positions, typically used for M-RoPE
    llama_pos x = 0;
    llama_pos y = 0;

    // token id at this cell; used by PLE n-gram hashing
    llama_token tok = LLAMA_TOKEN_NULL;

    // return true if the current 2D spatial position is greater than other
    bool is_2d_gt(llama_pos ox, llama_pos oy) const {
        return (y > oy) || (y == oy && x > ox);
    }

    void reset() {
        static_assert(std::is_trivially_copyable_v<llama_kv_cell_ext>);

        memset(this, 0, sizeof(*this));
    }
};

// compaction: a single contiguous run of cells to move from src to dst
struct kv_pack_move {
    uint32_t src_begin;  // source start index (old physical)
    uint32_t dst_begin;  // destination start index (new physical)
    uint32_t length;     // number of consecutive cells to move
};

// compaction: a plan that packs all used cells to [0, retained_count)
// all moves satisfy dst <= src, enabling in-place downward packing
struct kv_pack_plan {
    std::vector<kv_pack_move> moves;  // sorted by src_begin, all dst <= src
    uint32_t retained_count = 0;      // total cells after packing
};

// meta information about KV cells that can be part of multiple sequences at the same time
// TODO: add unit tests
class llama_kv_cells {
private:
    // FlashPrefill derived-cache invalidation stamp. Plain integers only:
    // no allocations, no callbacks, no container. Lazy: while tracking is
    // disabled the stamp stays 0 and no atomic/sync work happens (OFF exact
    // legacy behavior). The atomic counter is touched only when tracking is
    // enabled (first enable, enabled copy) — never on the disabled path.
    uint64_t gen_stamp = 0;
    uint64_t gen_val = 0;
    bool gen_enabled = false;

    static uint64_t new_stamp() {
        static std::atomic<uint64_t> next{1};
        const uint64_t s = next.fetch_add(1, std::memory_order_relaxed);
        // 0 is reserved as never-tracked. Wrap needs 2^64 stamps (practically
        // impossible); fail closed by saturating at MAX instead of reusing 0.
        // Consumers MUST treat stamp == MAX as always-invalid, like
        // generation == MAX.
        if (s == 0) {
            return std::numeric_limits<uint64_t>::max();
        }
        return s;
    }

    void bump_gen() {
        if (!gen_enabled) {
            return;
        }
        if (gen_val == std::numeric_limits<uint64_t>::max()) {
            return; // sticky saturated: consumer must treat as always-dirty
        }
        ++gen_val;
    }

public:
    using seq_set_t = std::bitset<LLAMA_MAX_SEQ>;

    // FlashPrefill derived-cache invalidation stamp (opt-in, OFF by default).
    //
    // Contract for the FlashPrefill cache agent (pool/plan cache):
    //  - Disabled by default with stamp 0 / generation 0. While disabled no
    //    mutation bumps anything and no atomic/sync work happens (default
    //    construction, copy, and assignment of disabled objects never touch
    //    the stamp counter). No extra allocations and no behavior changes.
    //  - Enable with set_generation_enabled(true). The first enable assigns
    //    a unique stamp when none is held (stamp == 0) and bumps once
    //    (0->1) so the enabled baseline never equals the disabled 0.
    //    Disabling preserves stamp+generation (no reset); re-enabling bumps
    //    again so a new baseline cannot equal a possibly cached pre-disable
    //    baseline. Enabling an already-enabled object is idempotent.
    //  - While enabled, every successful mutation that can affect membership,
    //    positions, ext, shift, restore/copy-in, pack, rerot tag/publish/
    //    reclassify/freeze, or clearing bumps the generation (saturating).
    //    Read-only queries and failed/no-op operations do not bump, except
    //    where conservatively documented per method below.
    //  - Identity is (stamp, generation). An enabled copy (copy ctor/assign,
    //    cp() snapshot) gets a fresh stamp and inherits the source
    //    generation, so (stamp, generation) can never collide with the
    //    source. A disabled copy stays stamp 0 / generation 0 with no atomic
    //    work. Moves transfer (stamp, generation, enabled) with ordinary
    //    data-move semantics and clear the source to disabled 0/0/false with
    //    no atomic work, so two live objects never share a stamp.
    //    Clearing (reset/resize) preserves enabled+stamp and bumps instead
    //    of resetting to 0.
    //  - Overflow is fail-closed: generation saturates at UINT64_MAX and
    //    never wraps; stamp saturation also sticks at MAX. Consumers MUST
    //    treat generation == MAX or stamp == MAX as always-dirty (never
    //    reuse a cached derived entry keyed on either).
    //  - Composite ops (notably rerot_freeze_to_archive via seq_rm) may
    //    advance the generation by more than one. Consumers MUST compare for
    //    inequality, never for an exact +1 delta.
    //  - No lifecycle callbacks, no separate container: the owner only
    //    exposes this monotonic stamp. Derived caches poll (stamp,
    //    generation) and rebuild on change.
    llama_kv_cells() = default;

    llama_kv_cells(const llama_kv_cells & other)
        : gen_stamp(other.gen_enabled ? new_stamp() : 0),
          gen_val(other.gen_enabled ? other.gen_val : 0),
          gen_enabled(other.gen_enabled),
          has_shift(other.has_shift),
          used(other.used),
          pos(other.pos),
          ext(other.ext),
          shift(other.shift),
          seq(other.seq),
          rerot(other.rerot) {
        for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            seq_pos[s]  = other.seq_pos[s];
            seq_used[s] = other.seq_used[s];
        }
    }

    llama_kv_cells(llama_kv_cells && other) noexcept
        : gen_stamp(other.gen_stamp),
          gen_val(other.gen_val),
          gen_enabled(other.gen_enabled),
          has_shift(other.has_shift),
          used(std::move(other.used)),
          pos(std::move(other.pos)),
          ext(std::move(other.ext)),
          shift(std::move(other.shift)),
          seq(std::move(other.seq)),
          rerot(std::move(other.rerot)) {
        for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            seq_pos[s]  = std::move(other.seq_pos[s]);
            seq_used[s] = other.seq_used[s];
        }
        // Ordinary transfer, then clear the source with no atomic work so
        // two live objects never share a stamp.
        other.gen_stamp   = 0;
        other.gen_val     = 0;
        other.gen_enabled = false;
    }

    llama_kv_cells & operator=(const llama_kv_cells & other) {
        if (this != &other) {
            has_shift = other.has_shift;
            used      = other.used;
            pos       = other.pos;
            ext       = other.ext;
            shift     = other.shift;
            seq       = other.seq;
            rerot     = other.rerot;
            for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
                seq_pos[s]  = other.seq_pos[s];
                seq_used[s] = other.seq_used[s];
            }
            // Fresh stamp only when tracking; disabled copies stay 0/0/false
            // with no atomic work. The old (stamp, generation) can never
            // reappear, so derived caches conservatively miss.
            if (other.gen_enabled) {
                gen_stamp   = new_stamp();
                gen_val     = other.gen_val;
                gen_enabled = true;
            } else {
                gen_stamp   = 0;
                gen_val     = 0;
                gen_enabled = false;
            }
        }
        return *this;
    }

    llama_kv_cells & operator=(llama_kv_cells && other) noexcept {
        if (this != &other) {
            has_shift = other.has_shift;
            used      = std::move(other.used);
            pos       = std::move(other.pos);
            ext       = std::move(other.ext);
            shift     = std::move(other.shift);
            seq       = std::move(other.seq);
            rerot     = std::move(other.rerot);
            for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
                seq_pos[s]  = std::move(other.seq_pos[s]);
                seq_used[s] = other.seq_used[s];
            }
            gen_stamp   = other.gen_stamp;
            gen_val     = other.gen_val;
            gen_enabled = other.gen_enabled;
            other.gen_stamp   = 0;
            other.gen_val     = 0;
            other.gen_enabled = false;
        }
        return *this;
    }

    bool get_generation_enabled() const {
        return gen_enabled;
    }

    // Current mutation generation. Only meaningful when
    // get_generation_enabled() is true. 0 is the disabled/untracked value;
    // UINT64_MAX is sticky saturated and means always-dirty (never reuse).
    uint64_t get_generation() const {
        return gen_val;
    }

    // Invalidation identity. 0 means never-tracked (disabled, no atomic work
    // ever spent on this object). Non-zero values are unique per tracked
    // lifetime; MAX is sticky saturated and means always-invalid.
    // Always pair with get_generation().
    uint64_t get_generation_stamp() const {
        return gen_stamp;
    }

    void set_generation_enabled(bool enable) {
        if (enable == gen_enabled) {
            return;
        }
        gen_enabled = enable;
        if (enable) {
            // Lazy identity: unique stamp only when none is held, so the
            // disabled path never pays for an atomic.
            if (gen_stamp == 0) {
                gen_stamp = new_stamp();
            }
            // New observable baseline: never equal the disabled 0 nor a
            // possibly cached pre-disable baseline.
            if (gen_val != std::numeric_limits<uint64_t>::max()) {
                ++gen_val;
            }
        }
        // Disabling preserves stamp+generation (no reset to collision value).
    }

    void reset() {
        if (payload_id.size() < pos.size()) {
            payload_id.resize(pos.size(), 0);
            storage_generation.resize(pos.size(), 0);
        }
        for (uint32_t i = 0; i < pos.size(); ++i) {
            pos[i]   = -1;
            ext[i].reset();
            shift[i] =  0;
            seq[i].reset();
            rerot[i].reset();
            if (i < payload_id.size()) {
                payload_id[i] = 0;
                storage_generation[i] = 0;
            }
        }

        has_shift = false;

        used.clear();

        for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            seq_pos[s].clear();
            seq_used[s] = 0;
        }

        // Clearing is an epoch boundary (slot/seq reuse). Conservatively bump
        // even when already empty; never reset generation to 0.
        bump_gen();
    }

    void reset_shift() {
        const bool had_shift = has_shift;
        has_shift = false;

        for (uint32_t i = 0; i < shift.size(); ++i) {
            shift[i] = 0;
        }

        // No-op call (no pending shift) does not bump.
        if (had_shift) {
            bump_gen();
        }
    }

    uint32_t size() const {
        return pos.size();
    }

    void resize(uint32_t n) {
        pos.resize(n);
        ext.resize(n);
        shift.resize(n);
        seq.resize(n);
        rerot.resize(n);
        payload_id.resize(n, 0);
        storage_generation.resize(n, 0);

        // Single invalidation via reset(). Preserves enabled+stamp, bumps.
        reset();
    }

    bool is_empty(uint32_t i) const {
        assert(i < pos.size());
        assert((pos[i] < 0 && pos[i] == -1) || pos[i] >= 0);

        return pos[i] == -1;
    }

    uint32_t get_used() const {
        return used.size();
    }

    // the index of the first cell that is used
    // return 0 if no cells are used
    uint32_t used_min() const {
        return used.empty() ? 0 : *used.begin();
    }

    // the index of the last cell that is used + 1
    // return 0 if no cells are used
    uint32_t used_max_p1() const {
        return used.empty() ? 0 : *used.rbegin() + 1;
    }

    bool get_has_shift() const {
        return has_shift;
    }

    // move cell isrc to idst (used during defrag)
    //void mv(uint32_t isrc, uint32_t idst) {
    //    assert(isrc < pos.size());
    //    assert(idst < pos.size());

    //    assert(pos[idst] == -1);
    //    assert(pos[isrc] != -1);

    //    pos  [idst] = pos  [isrc];
    //    shift[idst] = shift[isrc];
    //    seq  [idst] = seq  [isrc];

    //    pos  [isrc] = -1;
    //    shift[isrc] =  0;
    //    seq  [isrc].reset();

    //    used.erase (isrc);
    //    used.insert(idst);
    //}

    // copy the state of cells [i, i + n) (used for save/restore the state of the cells)
    // Read-only on the source: never bumps the source. An enabled snapshot
    // gets a fresh stamp and inherits the source baseline; a disabled snapshot
    // stays stamp 0 with no atomic work.
    llama_kv_cells cp(uint32_t i, uint32_t n) const {
        assert(i + n <= pos.size());

        llama_kv_cells res;

        res.resize(n);

        for (uint32_t j = 0; j < n; ++j) {
            const auto idx = i + j;

            res.pos[j] = pos[idx];
            res.ext[j] = ext[idx];
            res.seq[j] = seq[idx];
            res.rerot[j] = rerot[idx];
            res.payload_id[j] = idx < payload_id.size() ? payload_id[idx] : 0;
            res.storage_generation[j] = idx < storage_generation.size() ? storage_generation[idx] : 0;

            assert(shift[idx] == 0);
        }

        if (gen_enabled) {
            res.gen_stamp   = new_stamp();
            res.gen_val     = gen_val;
            res.gen_enabled = true;
        }

        return res;
    }

    // copy the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
    // Same lazy snapshot contract as cp(i, n).
    llama_kv_cells cp(const std::vector<uint32_t> & idxs) const {
        llama_kv_cells res;

        res.resize(idxs.size());

        for (uint32_t j = 0; j < idxs.size(); ++j) {
            const auto idx = idxs[j];

            res.pos[j] = pos[idx];
            res.ext[j] = ext[idx];
            res.seq[j] = seq[idx];
            res.rerot[j] = rerot[idx];
            res.payload_id[j] = idx < payload_id.size() ? payload_id[idx] : 0;
            res.storage_generation[j] = idx < storage_generation.size() ? storage_generation[idx] : 0;

            assert(shift[idx] == 0);
        }

        if (gen_enabled) {
            res.gen_stamp   = new_stamp();
            res.gen_val     = gen_val;
            res.gen_enabled = true;
        }

        return res;
    }

    // set the state of cells [i, i + other.pos.size()) (used for save/restore the state of the cells)
    // Restore path: conservatively bumps once when enabled, even if the
    // copied bytes happen to be identical. Empty restore is a no-op.
    void set(uint32_t i, const llama_kv_cells & other) {
        assert(i + other.pos.size() <= pos.size());

        if (payload_id.size() < pos.size()) {
            payload_id.resize(pos.size(), 0);
            storage_generation.resize(pos.size(), 0);
        }
        if (other.pos.empty()) {
            return;
        }

        for (uint32_t j = 0; j < other.pos.size(); ++j) {
            const auto idx = i + j;

            if (pos[idx] == -1 && other.pos[j] != -1) {
                used.insert(i + j);
            }

            if (pos[idx] != -1 && other.pos[j] == -1) {
                used.erase(i + j);
            }

            if (pos[idx] != -1) {
                seq_pos_rm(i + j);
            }

            pos[idx] = other.pos[j];
            ext[idx] = other.ext[j];
            seq[idx] = other.seq[j];
            rerot[idx] = other.rerot[j];
            payload_id[idx] = j < other.payload_id.size() ? other.payload_id[j] : 0;
            storage_generation[idx] = j < other.storage_generation.size() ? other.storage_generation[j] : 0;

            if (pos[idx] != -1) {
                seq_pos_add(i + j);
            }

            assert(shift[idx] == 0);
        }

        bump_gen();
    }

    // set the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
    // Same conservative single-bump contract as set(i, other).
    void set(const std::vector<uint32_t> & idxs, const llama_kv_cells & other) {
        assert(idxs.size() == other.pos.size());

        if (payload_id.size() < pos.size()) {
            payload_id.resize(pos.size(), 0);
            storage_generation.resize(pos.size(), 0);
        }
        if (idxs.empty()) {
            return;
        }

        for (uint32_t j = 0; j < other.pos.size(); ++j) {
            const auto idx = idxs[j];

            if (pos[idx] == -1 && other.pos[j] != -1) {
                used.insert(idx);
            }

            if (pos[idx] != -1 && other.pos[j] == -1) {
                used.erase(idx);
            }

            if (pos[idx] != -1) {
                seq_pos_rm(idx);
            }

            pos[idx] = other.pos[j];
            ext[idx] = other.ext[j];
            seq[idx] = other.seq[j];
            rerot[idx] = other.rerot[j];
            payload_id[idx] = j < other.payload_id.size() ? other.payload_id[j] : 0;
            storage_generation[idx] = j < other.storage_generation.size() ? other.storage_generation[j] : 0;

            if (pos[idx] != -1) {
                seq_pos_add(idx);
            }

            assert(shift[idx] == 0);
        }

        bump_gen();
    }

    // clear a non-empty cell
    void rm(uint32_t i) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        seq_pos_rm(i);
        seq[i].reset();

        pos[i] = -1;
        ext[i].reset();
        shift[i] = 0;
        rerot[i].reset();
        if (i < payload_id.size()) {
            payload_id[i] = 0;
            storage_generation[i] = 0;
        }

        used.erase(i);

        bump_gen();
    }

    // Generate a compaction plan that packs all used cells to [0, used)
    // All moves satisfy dst <= src (in-place downward)
    // Read-only: never bumps. Cache the plan only under the current
    // (stamp, generation); re-plan after any bump.
    kv_pack_plan make_pack_plan() const {
        kv_pack_plan plan;

        uint32_t dst = 0;
        auto it = used.begin();
        while (it != used.end()) {
            const uint32_t src_begin = *it;
            uint32_t length = 1;
            ++it;
            while (it != used.end() && *it == src_begin + length) {
                ++length;
                ++it;
            }

            plan.moves.push_back({src_begin, dst, length});
            dst += length;
        }

        plan.retained_count = dst;
        return plan;
    }

    // Execute a pack plan: move cells according to the plan
    // After execution, used indices are [0, retained_count)
    // This only moves metadata (pos, ext, seq, shift). K/V data is moved separately.
    // Empty plan is a no-op and does not bump. Any non-empty plan bumps once,
    // including an all-identity plan (conservative: physical rewrite happened).
    void apply_pack(const kv_pack_plan & plan) {
        if (plan.moves.empty()) {
            return;
        }

        if (payload_id.size() < pos.size()) {
            payload_id.resize(pos.size(), 0);
            storage_generation.resize(pos.size(), 0);
        }

        const uint32_t n = plan.retained_count;

        // Save metadata for all retained cells in ascending source order
        std::vector<llama_pos>         saved_pos  (n);
        std::vector<llama_kv_cell_ext> saved_ext  (n);
        std::vector<seq_set_t>         saved_seq  (n);
        std::vector<llama_pos>         saved_shift(n);
        std::vector<llama_kv_rerot_meta> saved_rerot(n);
        std::vector<uint64_t>          saved_payload_id(n);
        std::vector<uint64_t>          saved_storage_gen(n);

        uint32_t dst = 0;
        for (const auto & move : plan.moves) {
            for (uint32_t k = 0; k < move.length; ++k) {
                const uint32_t src = move.src_begin + k;
                saved_pos  [dst] = pos  [src];
                saved_ext  [dst] = ext  [src];
                saved_seq  [dst] = seq  [src];
                saved_shift[dst] = shift[src];
                saved_rerot[dst] = rerot[src];
                saved_payload_id [dst] = src < payload_id.size() ? payload_id[src] : 0;
                saved_storage_gen[dst] = src < storage_generation.size() ? storage_generation[src] : 0;
                ++dst;
            }
        }

        // Clear all old used cells (seq_pos, seq, pos, ext, shift, used set)
        for (const uint32_t idx : used) {
            seq_pos_rm(idx);
            seq  [idx].reset();
            pos  [idx] = -1;
            ext  [idx].reset();
            shift[idx] =  0;
            rerot[idx].reset();
            if (idx < payload_id.size()) {
                payload_id[idx] = 0;
                storage_generation[idx] = 0;
            }
        }
        used.clear();

        // Write metadata to new dense positions [0, n)
        for (uint32_t i = 0; i < n; ++i) {
            pos  [i] = saved_pos  [i];
            ext  [i] = saved_ext  [i];
            seq  [i] = saved_seq  [i];
            shift[i] = saved_shift[i];
            rerot[i] = saved_rerot[i];
            payload_id[i] = saved_payload_id[i];
            storage_generation[i] = saved_storage_gen[i];
            used.insert(i);
            seq_pos_add(i);
        }

        bump_gen();
    }

    // note: call only if the cell has seq_id
    // return true if the cell becomes empty
    // Always a real membership change: always bumps when enabled.
    bool seq_rm(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());
        assert(seq[i].test(seq_id));
        assert(pos[i] != -1);
        assert(seq_id >= 0);

        seq[i].reset(seq_id);
        seq_pos_dec(seq_id, pos[i]);

        bump_gen();

        if (seq[i].none()) {
            pos[i] = -1;
            ext[i].reset();
            shift[i] = 0;
            rerot[i].reset();
            if (i < payload_id.size()) {
                payload_id[i] = 0;
                storage_generation[i] = 0;
            }

            used.erase(i);

            return true;
        }

        return false;
    }

    // return true if the cell becomes empty (i.e. it did not contain seq_id before the call)
    // Bumps only on a real change: keeping the sole existing ref or filtering
    // an already-empty cell does not bump; removing other refs or freeing the
    // cell bumps.
    bool seq_keep(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());

        if (seq[i].test(seq_id)) {
            // Extra bitset count only when tracking: keeps OFF path exact.
            const bool changed = gen_enabled && seq[i].count() != 1;
            seq_pos_rm(i);
            seq[i].reset();

            seq[i].set(seq_id);
            seq_pos_inc(seq_id, pos[i]);

            if (changed) {
                bump_gen();
            }

            return false;
        }

        if (seq[i].any()) {
            seq_pos_rm(i);
            seq[i].reset();

            pos[i] = -1;
            ext[i].reset();
            shift[i] = 0;
            rerot[i].reset();
            if (i < payload_id.size()) {
                payload_id[i] = 0;
                storage_generation[i] = 0;
            }

            used.erase(i);

            bump_gen();

            return true;
        }

        assert(pos[i] == -1);

        return false;
    }

    // number of different sequences in the cell
    int seq_count(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return seq[i].count();
    }

    // the full set of sequences this cell is visible to
    const seq_set_t & seq_get_all(uint32_t i) const {
        assert(i < pos.size());

        return seq[i];
    }

    // check if the cell contains seq_id
    bool seq_has(uint32_t i, llama_seq_id seq_id) const {
        assert(i < pos.size());
        assert(seq_id >= 0);

        return seq[i].test(seq_id);
    }

    // note: call only if the cell is not empty and the seq_id is not in the cell
    // Always a new reference: always bumps when enabled.
    void seq_add(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());
        assert(pos[i] != -1);
        assert(!seq[i].test(seq_id));

        seq[i].set(seq_id);
        seq_pos_inc(seq_id, pos[i]);

        bump_gen();
    }

    // the token of the cell of sequence seq_id at the largest position <= p
    // when several cells share that position, the one with the highest index wins
    // return LLAMA_TOKEN_NULL if the sequence has no cell at or before p
    llama_token seq_pos_tok_le(llama_seq_id seq_id, llama_pos p) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        llama_token best_tok = LLAMA_TOKEN_NULL;
        llama_pos best_pos = -1;
        uint32_t best_idx = 0;

        // If used cells are recorded, only iterate over used cells rather than full capacity!
        if (!used.empty()) {
            for (auto it = used.begin(); it != used.end(); ++it) {
                uint32_t i = *it;
                if (pos[i] < 0 || !seq[i].test(seq_id) || pos[i] > p) {
                    continue;
                }
                if (pos[i] > best_pos || (pos[i] == best_pos && i >= best_idx)) {
                    best_pos = pos[i];
                    best_idx = i;
                    best_tok = ext[i].tok;
                }
            }
            return best_tok;
        }

        for (uint32_t i = 0; i < pos.size(); ++i) {
            if (pos[i] < 0 || !seq[i].test(seq_id) || pos[i] > p) {
                continue;
            }
            if (pos[i] > best_pos || (pos[i] == best_pos && i >= best_idx)) {
                best_pos = pos[i];
                best_idx = i;
                best_tok = ext[i].tok;
            }
        }

        return best_tok;
    }

    // return the sequence id of this cell
    // note: call only for cells with exactly one sequence
    llama_seq_id seq_get(uint32_t i) const {
        assert(seq[i].count() == 1);

        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (seq[i].test(s)) {
                return s;
            }
        }

        return -1;
    }

    // Exact full multi-ref seq snapshot for overwrite-victim save/restore.
    // Unlike seq_get (single-ref only), this preserves every reference so a
    // restored cell is bit-identical in sequence membership.
    using seq_snapshot_t = std::bitset<LLAMA_MAX_SEQ>;

    seq_snapshot_t seq_snapshot(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return seq[i];
    }

    // Restore a previously snapshotted seq set onto an empty cell whose pos
    // is already set (e.g. after rm + pos_set). Rebuilds seq_pos bookkeeping
    // exactly as seq_add would for each member.
    void seq_restore(uint32_t i, const seq_snapshot_t & set) {
        assert(i < pos.size());
        assert(pos[i] != -1);
        assert(seq[i].none());
        assert(!set.none());

        seq[i] = set;
        seq_pos_add(i);
    }

    // the minimum position of sequence seq_id currently present in any of the cells
    // return -1 if the sequence is not present
    llama_pos seq_pos_min(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        if (seq_pos[seq_id].empty()) {
            return -1;
        }

        assert(seq_pos[seq_id].begin()->second > 0);

        return seq_pos[seq_id].begin()->first;
    }

    // the maximum position of sequence seq_id currently present in any of the cells
    // return -1 if the sequence is not present
    llama_pos seq_pos_max(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        if (seq_pos[seq_id].empty()) {
            return -1;
        }

        assert(seq_pos[seq_id].rbegin()->second > 0);

        return seq_pos[seq_id].rbegin()->first;
    }

    uint32_t seq_get_used(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        return seq_used[seq_id];
    }

    // note: call only if the cell is not empty
    llama_pos pos_get(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return pos[i];
    }

    const llama_kv_cell_ext & ext_get(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return ext[i];
    }

    const llama_kv_rerot_meta & rerot_get(uint32_t i) const {
        assert(i < rerot.size());
        return rerot[i];
    }

    // Attach a complete RERoT write tag to a non-empty cell. A zero episode id
    // is reserved for ordinary KV and therefore clears the metadata.
    // Conservative: always bumps when enabled, even if the tag is identical.
    void rerot_set(uint32_t i, llama_kv_rerot_meta meta) {
        assert(i < rerot.size());
        assert(pos[i] != -1);

        if (!meta.active()) {
            meta.reset();
        } else {
            assert(meta.node_id != LLAMA_REROT_NODE_INVALID);
            assert(meta.run_id != LLAMA_REROT_RUN_INVALID);
            assert(meta.visibility != llama_rerot_visibility::normal);
            if (meta.visibility == llama_rerot_visibility::pending_record) {
                assert(meta.publish_epoch == 0);
            }
        }
        rerot[i] = meta;

        bump_gen();
    }

    // Bumps only when the cell actually carried an active tag; resetting an
    // already-inactive cell is a no-op and does not bump.
    void rerot_reset(uint32_t i) {
        assert(i < rerot.size());
        const bool was_active = gen_enabled && rerot[i].active();
        rerot[i].reset();
        if (was_active) {
            bump_gen();
        }
    }

    // Bumps only on success (true). Failed guards leave state untouched and
    // do not bump.
    bool rerot_publish(
            uint32_t i,
            uint64_t episode_id,
            llama_rerot_run_id run_id,
            uint64_t publish_epoch) {
        assert(i < rerot.size());

        auto & meta = rerot[i];
        if (pos[i] == -1 || publish_epoch == 0 || meta.episode_id != episode_id ||
            meta.run_id != run_id || meta.visibility != llama_rerot_visibility::pending_record) {
            return false;
        }

        meta.visibility = llama_rerot_visibility::public_live;
        meta.publish_epoch = publish_epoch;

        bump_gen();

        return true;
    }

    // Bumps only on success (true). Failed guards leave state untouched and
    // do not bump.
    bool rerot_reclassify(
            uint32_t i,
            uint64_t episode_id,
            llama_rerot_run_id run_id,
            llama_rerot_visibility expected,
            llama_rerot_visibility replacement,
            uint64_t publish_epoch) {
        assert(i < rerot.size());

        auto & meta = rerot[i];
        if (pos[i] == -1 || expected == llama_rerot_visibility::normal ||
            replacement == llama_rerot_visibility::normal || meta.episode_id != episode_id ||
            meta.run_id != run_id || meta.visibility != expected) {
            return false;
        }
        if ((replacement == llama_rerot_visibility::public_live && publish_epoch == 0) ||
            (replacement != llama_rerot_visibility::public_live && publish_epoch != 0)) {
            return false;
        }

        meta.visibility = replacement;
        meta.publish_epoch = replacement == llama_rerot_visibility::public_live ? publish_epoch : 0;

        bump_gen();

        return true;
    }

    // Physical lookup of one logical run within this stream. Appends the
    // physical indices of every resident (non-empty) cell whose RERoT
    // metadata matches (episode_id, run_id), in ascending cell order.
    // Returns the number of appended indices. This is the only sanctioned
    // way to translate a stable run id to physical locations; callers must
    // never cache the result across compaction, eviction, or restore.
    // Read-only on the owner: never bumps. Callers key derived caches on
    // (stamp, generation), never on collect results.
    uint32_t rerot_collect_run(
            uint64_t episode_id,
            llama_rerot_run_id run_id,
            std::vector<uint32_t> & out) const {
        if (episode_id == 0 || run_id == LLAMA_REROT_RUN_INVALID) {
            return 0;
        }

        uint32_t count = 0;
        for (const uint32_t i : used) {
            const auto & meta = rerot[i];
            if (meta.episode_id == episode_id && meta.run_id == run_id) {
                out.push_back(i);
                ++count;
            }
        }
        return count;
    }

    // True when any resident cell carries RERoT metadata. Used by the v1
    // persistence guard: such classification has no stable serialized form.
    bool rerot_has_active() const {
        for (const uint32_t i : used) {
            if (rerot[i].active()) {
                return true;
            }
        }
        return false;
    }

    // True when any resident cell referenced by seq_id carries RERoT metadata.
    bool rerot_has_active_seq(llama_seq_id seq_id) const {
        assert(seq_id >= 0 && seq_id < LLAMA_MAX_SEQ);

        for (const uint32_t i : used) {
            if (rerot[i].active() && seq[i].test(seq_id)) {
                return true;
            }
        }
        return false;
    }

    // Attention-only archive freeze of one execution sequence (§7.2). Every
    // resident cell referenced by exec_seq is visited: PUBLIC_LIVE cells of
    // episode_id gain archive_seq as a keeper ref (when not already present)
    // so their K/V survives; PRIVATE/PENDING/ordinary cells gain no new ref.
    // All exec_seq refs are then removed, freeing the execution handle
    // without moving any K/V data or touching visibility metadata. Returns
    // the number of public cells now kept alive by archive_seq.
    // Note: ordinary shared-prefix cells survive only when archive_seq
    // already references them (the caller archives the serial prefix
    // separately); cells left without any seq ref are freed, which is the
    // intended release for private/pending history.
    // Generation: empty touched set is a no-op (no bump). Otherwise bumps via
    // the inner seq_rm calls (once per touched cell) plus one final bump, so
    // the advance may exceed +1; compare for inequality, not exact delta.
    size_t rerot_freeze_to_archive(
            uint64_t episode_id,
            llama_seq_id exec_seq,
            llama_seq_id archive_seq) {
        assert(episode_id != 0);
        assert(exec_seq >= 0 && exec_seq < LLAMA_MAX_SEQ);
        assert(archive_seq >= 0 && archive_seq < LLAMA_MAX_SEQ);
        assert(exec_seq != archive_seq);

        std::vector<uint32_t> touched;
        for (const uint32_t i : used) {
            if (seq[i].test(exec_seq)) {
                touched.push_back(i);
            }
        }

        if (touched.empty()) {
            return 0;
        }

        size_t kept = 0;
        for (const uint32_t i : touched) {
            const auto & meta = rerot[i];
            const bool keep = meta.active() && meta.episode_id == episode_id &&
                meta.visibility == llama_rerot_visibility::public_live &&
                meta.publish_epoch != 0;

            if (keep) {
                if (!seq[i].test(archive_seq)) {
                    seq[i].set(archive_seq);
                    seq_pos_inc(archive_seq, pos[i]);
                }
                ++kept;
            }

            seq_rm(i, exec_seq);
        }

        bump_gen();

        return kept;
    }

    // note: call only if the cell is not empty
    llama_pos get_shift(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return shift[i];
    }

    // check if a cell is not empty and its position is within [p0, p1)
    bool pos_in(uint32_t i, llama_pos p0, llama_pos p1) const {
        assert(i < pos.size());

        return pos[i] >= p0 && pos[i] < p1;
    }

    // set the position of an empty cell
    // does not modify "has_shift"
    // note: call only if the cell is empty
    // Always a membership change: always bumps when enabled.
    void pos_set(uint32_t i, llama_pos p) {
        assert(i < pos.size());
        assert(pos[i] == -1);
        assert(seq[i].none());

        if (payload_id.size() < pos.size()) {
            payload_id.resize(pos.size(), 0);
            storage_generation.resize(pos.size(), 0);
        }

        pos[i] = p;
        rerot[i].reset();
        payload_id[i] = allocate_payload_id();
        storage_generation[i] = 1;

        used.insert(i);

        bump_gen();
    }

    // Conservative: always bumps when enabled, even if the value is identical.
    void ext_set(uint32_t i, llama_kv_cell_ext p) {
        assert(i < ext.size());
        ext[i] = p;

        bump_gen();
    }

    uint64_t payload_id_get(uint32_t i) const {
        return i < payload_id.size() ? payload_id[i] : 0;
    }

    uint64_t storage_generation_get(uint32_t i) const {
        return i < storage_generation.size() ? storage_generation[i] : 0;
    }

    void payload_id_set(uint32_t i, uint64_t pid, uint64_t gen = 1) {
        if (i >= payload_id.size()) {
            payload_id.resize(std::max<size_t>(i + 1, pos.size()), 0);
            storage_generation.resize(std::max<size_t>(i + 1, pos.size()), 0);
        }
        payload_id[i] = pid;
        storage_generation[i] = gen;
    }

    void storage_generation_inc(uint32_t i) {
        if (i < storage_generation.size()) {
            storage_generation[i]++;
        }
    }

    // Restore preflight: advance the process-wide payload counter past max_pid
    // so future allocate_payload_id() calls cannot collide with restored ids.
    // Only moves forward (CAS loop); fails closed with zero mutation when
    // max_pid is UINT64_MAX or the counter is already saturated (wrap can
    // never be represented). Monotonic under concurrency. Cells remain the
    // sole metadata owner: the counter lives here and is never reset.
    static bool reserve_payload_ids_through(uint64_t max_pid) {
        if (max_pid == UINT64_MAX) {
            return false;
        }
        std::atomic<uint64_t> & ctr = payload_id_counter();
        uint64_t cur = ctr.load(std::memory_order_relaxed);
        while (cur <= max_pid) {
            if (cur == UINT64_MAX) {
                return false;
            }
            // Target max_pid + 1 cannot wrap: max_pid < UINT64_MAX here.
            if (ctr.compare_exchange_weak(cur, max_pid + 1, std::memory_order_relaxed)) {
                return true;
            }
        }
        return true; // already past max_pid: nothing to do
    }

    // pos[i] = pos[i] + d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    // Always bumps when enabled, even for d == 0 (has_shift is still set).
    bool pos_add(uint32_t i, llama_pos d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        seq_pos_rm(i);

        pos[i]   += d;
        shift[i] += d;

        has_shift = true;

        bump_gen();

        if (pos[i] < 0) {
            seq[i].reset();
            pos[i] = -1;
            shift[i] = 0;
            rerot[i].reset();
            if (i < payload_id.size()) {
                payload_id[i] = 0;
                storage_generation[i] = 0;
            }

            used.erase(i);

            return true;
        }

        seq_pos_add(i);

        return false;
    }

    // pos[i] = pos[i] / d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    // Always bumps when enabled.
    void pos_div(uint32_t i, int d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        const llama_pos p_old = pos[i];

        seq_pos_rm(i);

        pos[i]   /= d;
        shift[i] += p_old - pos[i];

        seq_pos_add(i);

        has_shift = true;

        bump_gen();
    }

private:
    bool has_shift = false;

    // set of indices of used cells (i.e. pos[i] != -1, allowed to not have any seq_id)
    std::set<uint32_t> used;

    std::vector<llama_pos> pos;

    // stores extra info per cell
    std::vector<llama_kv_cell_ext> ext;

    // this array accumulates any applied shifts to the pos array since the last reset_shift() call
    // this is used to queue multiple updates to the pos array, which in the end can be applied in one go:
    //
    //   cells.pos_add(x, shift_x);
    //   cells.pos_div(y, shift_y);
    //   ...
    //
    //   if (cells.has_shift()) {
    //      for (int i = 0; i < n; ++i) {
    //          auto shift_i = cells.get_shift(i);
    //          ...
    //      }
    //      cells.reset_shift();
    //   }
    //
    std::vector<llama_pos> shift;


    // the bitset seq[i] tells us which sequences are currently occupying the i-th cell
    std::vector<seq_set_t> seq;

    // RERoT metadata is part of the physical cell lifecycle. It must never be
    // mirrored in server-side physical-index tables because TriAttention
    // compaction can move cells.
    std::vector<llama_kv_rerot_meta> rerot;

    // Stable payload ID and storage generation.
    // Owned exclusively by cells; compaction/moves preserve them; occupied cells have nonzero payload_id.
    std::vector<uint64_t> payload_id;
    std::vector<uint64_t> storage_generation;

    // Single process-wide payload counter backing allocate + reserve.
    // One accessor (no duplicate function-local statics) so reservation and
    // allocation can never observe different counters.
    static std::atomic<uint64_t> & payload_id_counter() {
        static std::atomic<uint64_t> ctr{1};
        return ctr;
    }

    static inline uint64_t allocate_payload_id() {
        uint64_t id = payload_id_counter().fetch_add(1, std::memory_order_relaxed);
        if (id == 0) {
            id = payload_id_counter().fetch_add(1, std::memory_order_relaxed);
        }
        return id;
    }

    // the set seq_pos[s][p] tells us how many times the position p is currently present for sequence s
    // if the position p is not present, seq_pos[s][p] is not set
    // this way seq_pos[s].begin() and seq_pos[s].rbegin() give us the min/max positions currently in the cache
    //
    // note that we cannot a use an std::set because in some cases a position can occur more than once for the same seq:
    //  - during performing a cache reuse via (rm + add)
    //  - some vision models have input embeddings with repeating positions
    //
    std::map<llama_pos, int> seq_pos[LLAMA_MAX_SEQ];
    uint32_t seq_used[LLAMA_MAX_SEQ] = {};

    // helper functions for updating `seq_pos`, once cell at a time:

    void seq_pos_dec(llama_seq_id s, llama_pos p) {
        auto it = seq_pos[s].find(p);
        assert(it != seq_pos[s].end());
        assert(seq_used[s] > 0);

        seq_used[s]--;
        if (--it->second == 0) {
            seq_pos[s].erase(it);
        }
    }

    void seq_pos_inc(llama_seq_id s, llama_pos p) {
        seq_pos[s][p]++;
        seq_used[s]++;
    }

    // remove cell i
    void seq_pos_rm(uint32_t i) {
        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (seq[i].test(s)) {
                seq_pos_dec(s, pos[i]);
            }
        }
    }

    // add cell i
    void seq_pos_add(uint32_t i) {
        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (seq[i].test(s)) {
                seq_pos_inc(s, pos[i]);
            }
        }
    }
};

using llama_kv_cells_vec = std::vector<llama_kv_cells>;
