#pragma once

#include "llama.h"
#include "llama-cparams.h"
#include "llama-rerot.h"

#include <bitset>
#include <cassert>
#include <cstring>
#include <map>
#include <set>
#include <vector>

struct llama_kv_cell_ext {
    // 2D spatial positions, typically used for M-RoPE
    llama_pos x = 0;
    llama_pos y = 0;

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
public:
    void reset() {
        for (uint32_t i = 0; i < pos.size(); ++i) {
            pos[i]   = -1;
            ext[i].reset();
            shift[i] =  0;
            seq[i].reset();
            rerot[i].reset();
        }

        has_shift = false;

        used.clear();

        for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            seq_pos[s].clear();
            seq_used[s] = 0;
        }
    }

    void reset_shift() {
        has_shift = false;

        for (uint32_t i = 0; i < shift.size(); ++i) {
            shift[i] = 0;
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

            assert(shift[idx] == 0);
        }

        return res;
    }

    // copy the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
    llama_kv_cells cp(const std::vector<uint32_t> & idxs) const {
        llama_kv_cells res;

        res.resize(idxs.size());

        for (uint32_t j = 0; j < idxs.size(); ++j) {
            const auto idx = idxs[j];

            res.pos[j] = pos[idx];
            res.ext[j] = ext[idx];
            res.seq[j] = seq[idx];
            res.rerot[j] = rerot[idx];

            assert(shift[idx] == 0);
        }

        return res;
    }

    // set the state of cells [i, i + other.pos.size()) (used for save/restore the state of the cells)
    void set(uint32_t i, const llama_kv_cells & other) {
        assert(i + other.pos.size() <= pos.size());

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

            if (pos[idx] != -1) {
                seq_pos_add(i + j);
            }

            assert(shift[idx] == 0);
        }
    }

    // set the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
    void set(const std::vector<uint32_t> & idxs, const llama_kv_cells & other) {
        assert(idxs.size() == other.pos.size());

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

            if (pos[idx] != -1) {
                seq_pos_add(idx);
            }

            assert(shift[idx] == 0);
        }
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

        used.erase(i);
    }

    // Generate a compaction plan that packs all used cells to [0, used)
    // All moves satisfy dst <= src (in-place downward)
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
    void apply_pack(const kv_pack_plan & plan) {
        if (plan.moves.empty()) {
            return;
        }

        const uint32_t n = plan.retained_count;

        // Save metadata for all retained cells in ascending source order
        std::vector<llama_pos>         saved_pos  (n);
        std::vector<llama_kv_cell_ext> saved_ext  (n);
        std::vector<seq_set_t>         saved_seq  (n);
        std::vector<llama_pos>         saved_shift(n);
        std::vector<llama_kv_rerot_meta> saved_rerot(n);

        uint32_t dst = 0;
        for (const auto & move : plan.moves) {
            for (uint32_t k = 0; k < move.length; ++k) {
                const uint32_t src = move.src_begin + k;
                saved_pos  [dst] = pos  [src];
                saved_ext  [dst] = ext  [src];
                saved_seq  [dst] = seq  [src];
                saved_shift[dst] = shift[src];
                saved_rerot[dst] = rerot[src];
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
        }
        used.clear();

        // Write metadata to new dense positions [0, n)
        for (uint32_t i = 0; i < n; ++i) {
            pos  [i] = saved_pos  [i];
            ext  [i] = saved_ext  [i];
            seq  [i] = saved_seq  [i];
            shift[i] = saved_shift[i];
            rerot[i] = saved_rerot[i];
            used.insert(i);
            seq_pos_add(i);
        }
    }

    // note: call only if the cell has seq_id
    // return true if the cell becomes empty
    bool seq_rm(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());
        assert(seq[i].test(seq_id));
        assert(pos[i] != -1);
        assert(seq_id >= 0);

        seq[i].reset(seq_id);
        seq_pos_dec(seq_id, pos[i]);

        if (seq[i].none()) {
            pos[i] = -1;
            ext[i].reset();
            shift[i] = 0;
            rerot[i].reset();

            used.erase(i);

            return true;
        }

        return false;
    }

    // return true if the cell becomes empty (i.e. it did not contain seq_id before the call)
    bool seq_keep(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());

        if (seq[i].test(seq_id)) {
            seq_pos_rm(i);
            seq[i].reset();

            seq[i].set(seq_id);
            seq_pos_inc(seq_id, pos[i]);

            return false;
        }

        if (seq[i].any()) {
            seq_pos_rm(i);
            seq[i].reset();

            pos[i] = -1;
            ext[i].reset();
            shift[i] = 0;
            rerot[i].reset();

            used.erase(i);

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

    // check if the cell contains seq_id
    bool seq_has(uint32_t i, llama_seq_id seq_id) const {
        assert(i < pos.size());
        assert(seq_id >= 0);

        return seq[i].test(seq_id);
    }

    // note: call only if the cell is not empty and the seq_id is not in the cell
    void seq_add(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());
        assert(pos[i] != -1);
        assert(!seq[i].test(seq_id));

        seq[i].set(seq_id);
        seq_pos_inc(seq_id, pos[i]);
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
    }

    void rerot_reset(uint32_t i) {
        assert(i < rerot.size());
        rerot[i].reset();
    }

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
        return true;
    }

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
        return true;
    }

    // Physical lookup of one logical run within this stream. Appends the
    // physical indices of every resident (non-empty) cell whose RERoT
    // metadata matches (episode_id, run_id), in ascending cell order.
    // Returns the number of appended indices. This is the only sanctioned
    // way to translate a stable run id to physical locations; callers must
    // never cache the result across compaction, eviction, or restore.
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
    void pos_set(uint32_t i, llama_pos p) {
        assert(i < pos.size());
        assert(pos[i] == -1);
        assert(seq[i].none());

        pos[i] = p;
        rerot[i].reset();

        used.insert(i);
    }

    void ext_set(uint32_t i, llama_kv_cell_ext p) {
        assert(i < ext.size());
        ext[i] = p;
    }

    // pos[i] = pos[i] + d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    bool pos_add(uint32_t i, llama_pos d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        seq_pos_rm(i);

        pos[i]   += d;
        shift[i] += d;

        has_shift = true;

        if (pos[i] < 0) {
            seq[i].reset();
            pos[i] = -1;
            shift[i] = 0;
            rerot[i].reset();

            used.erase(i);

            return true;
        }

        seq_pos_add(i);

        return false;
    }

    // pos[i] = pos[i] / d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    void pos_div(uint32_t i, int d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        const llama_pos p_old = pos[i];

        seq_pos_rm(i);

        pos[i]   /= d;
        shift[i] += p_old - pos[i];

        seq_pos_add(i);

        has_shift = true;
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

    using seq_set_t = std::bitset<LLAMA_MAX_SEQ>;

    // the bitset seq[i] tells us which sequences are currently occupying the i-th cell
    std::vector<seq_set_t> seq;

    // RERoT metadata is part of the physical cell lifecycle. It must never be
    // mirrored in server-side physical-index tables because TriAttention
    // compaction can move cells.
    std::vector<llama_kv_rerot_meta> rerot;

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
