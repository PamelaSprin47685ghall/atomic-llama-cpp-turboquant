#pragma once

#include "llama.h"
#include "llama-cparams.h"

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstring>
#include <map>
#include <set>
#include <utility>
#include <vector>

// Iterate only the set bits of a sequence mask. The common cell has one or two
// sequences; walking all LLAMA_MAX_SEQ bits on every non-trial journal record
// and every replay is the dominant cost of the reversible journal. The
// trial-record path still scans all bits. libstdc++ exposes the word scan; the
// portable fallback keeps the same behavior (ascending set bits) on other
// standard libraries.
template <typename F>
static inline void for_each_seq(const std::bitset<LLAMA_MAX_SEQ> & bs, F && f) {
#if defined(__GLIBCXX__)
    for (size_t s = bs._Find_first(); s < LLAMA_MAX_SEQ; s = bs._Find_next(s)) {
        f(static_cast<int>(s));
    }
#else
    for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (bs.test(s)) {
            f(s);
        }
    }
#endif
}

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
// Undo journal for the cells touched by a prepare trial. The prepare enables
// it on the affected streams. Every mutation records the cell's pre-trial
// state. When a mutation removes a used cell or a seq_pos entry, the journal
// extracts the existing node from the set/map and keeps it. The reverse
// replay re-inserts the owned nodes, so it never allocates and cannot fail.
// A generation mark deduplicates a cell.
class cell_journal {
public:
    using used_node_t    = std::set<uint32_t>::node_type;
    using seq_pos_node_t = std::map<llama_pos, int>::node_type;

    struct seq_node_t {
        llama_seq_id s = -1;
        seq_pos_node_t node;
    };

    // The common cell has one or two seq nodes. Keep them inside the journal
    // entry so a trial does not allocate a vector for every touched cell. The
    // rare wider cell falls back to the heap vector.
    class seq_nodes_t {
    public:
        seq_node_t * find(llama_seq_id s) {
            for (uint32_t k = 0; k < n_inl; ++k) {
                if (inl[k].s == s) {
                    return &inl[k];
                }
            }
            for (auto & e : more) {
                if (e.s == s) {
                    return &e;
                }
            }
            return nullptr;
        }

        const seq_node_t * find(llama_seq_id s) const {
            for (uint32_t k = 0; k < n_inl; ++k) {
                if (inl[k].s == s) {
                    return &inl[k];
                }
            }
            for (const auto & e : more) {
                if (e.s == s) {
                    return &e;
                }
            }
            return nullptr;
        }

        size_t size() const { return n_inl + more.size(); }
        size_t heap_capacity() const { return more.capacity(); }

        // n is the total node count the caller expects to hold.
        void reserve(size_t n) {
            if (n > 2) {
                more.reserve(n - 2);
            }
        }

        void emplace_back(llama_seq_id s, seq_pos_node_t && node) {
            if (n_inl < 2) {
                inl[n_inl].s = s;
                inl[n_inl].node = std::move(node);
                ++n_inl;
            } else {
                more.push_back(seq_node_t{ s, std::move(node) });
            }
        }

    private:
        seq_node_t inl[2];
        uint32_t n_inl = 0;
        std::vector<seq_node_t> more;
    };

    struct entry {
        uint32_t i;
        llama_pos pos;
        llama_kv_cell_ext ext;
        llama_pos shift;
        std::bitset<LLAMA_MAX_SEQ> seq;
        used_node_t used_node;
        seq_nodes_t seq_nodes;
    };

    // Per-trial ownership: the state at the trial snapshot and the nodes that
    // the trial removed from it. The normal restore uses this entry, so it
    // never allocates and never needs the pre-prepare entry.
    struct trial_entry {
        uint32_t i;
        uint32_t strm; // the cell stream this entry belongs to
        llama_pos pos;
        llama_kv_cell_ext ext;
        llama_pos shift;
        std::bitset<LLAMA_MAX_SEQ> seq;
        used_node_t used_node;
        seq_nodes_t seq_nodes;
    };

    struct trial_view {
        trial_entry * data = nullptr;
        size_t size = 0;

        size_t count() const { return size; }
        bool empty() const { return size == 0; }
        trial_entry * begin() const { return data; }
        trial_entry * end() const { return data + size; }
        trial_entry & operator[](size_t k) const { return data[k]; }
    };

    void begin(size_t n, uint32_t strm = 0) {
        if (seen_.size() < n) {
            seen_.resize(n, 0);
        }
        if (trial_seen_.size() < n) {
            trial_seen_.resize(n, 0);
        }

        ++gen_;
        if (gen_ == 0) {
            std::fill(seen_.begin(), seen_.end(), 0);
            gen_ = 1;
        }

        strm_ = strm;
        entries_.clear();
        discard_used_.clear();
        discard_seq_.clear();
        trial_entries_.clear();
        trial_ranges_.clear();
        if (entries_.capacity() < 64) {
            entries_.reserve(64);
        }
        if (trial_entries_.capacity() < 64) {
            trial_entries_.reserve(64);
        }
        if (trial_ranges_.capacity() < 8) {
            trial_ranges_.reserve(8);
        }
    }

    void begin_trial() {
        ++trial_gen_;
        if (trial_gen_ == 0) {
            std::fill(trial_seen_.begin(), trial_seen_.end(), 0);
            trial_gen_ = 1;
        }
        // every trial keeps its own stable range: the exception replay, the
        // normal restore and a later prepare may still take a node
        trial_ranges_.emplace_back(trial_entries_.size(), trial_entries_.size());
    }

    size_t trial_count() const { return trial_ranges_.size(); }

    trial_view trial_at(size_t k) {
        GGML_ASSERT(k < trial_ranges_.size());
        const size_t b = trial_ranges_[k].first;
        return { trial_entries_.data() + b, trial_ranges_[k].second - b };
    }

    uint32_t strm() const { return strm_; }

    static trial_entry * find_trial(trial_view v, uint32_t i) {
        for (auto & e : v) {
            if (e.i == i) {
                return &e;
            }
        }
        return nullptr;
    }

    entry * find(uint32_t i) {
        if (i >= seen_.size() || (seen_[i] >> 32) != gen_) {
            return nullptr;
        }
        return &entries_[seen_[i] & 0xFFFFFFFFu];
    }

    // Find the entry that still owns the node for the used set of cell i with
    // the recorded old position. The entry belongs to this stream: a node of
    // the same cell/position in another stream is a different cell.
    bool find_used_source(uint32_t i, llama_pos pos, entry ** ge_out, trial_entry ** te_out) {
        *ge_out = nullptr;
        *te_out = nullptr;

        entry * ge = find(i);
        if (ge != nullptr && ge->pos == pos && !ge->used_node.empty()) {
            *ge_out = ge;
            return true;
        }

        for (auto it = trial_entries_.rbegin(); it != trial_entries_.rend(); ++it) {
            trial_entry & te = *it;
            if (te.strm == strm_ && te.i == i && te.pos == pos && !te.used_node.empty()) {
                *te_out = &te;
                return true;
            }
        }

        return false;
    }

    bool find_seq_source(uint32_t i, llama_pos pos, llama_seq_id s, entry ** ge_out, trial_entry ** te_out) {
        *ge_out = nullptr;
        *te_out = nullptr;

        entry * ge = find(i);
        if (ge != nullptr && ge->pos == pos && ge->seq.test(s)) {
            seq_node_t * n = ge->seq_nodes.find(s);
            if (n != nullptr && !n->node.empty()) {
                *ge_out = ge;
                return true;
            }
        }

        // the trial entries are the common source: the trial's later mutations
        // and the lazy trial entry. scan them newest first.
        for (auto it = trial_entries_.rbegin(); it != trial_entries_.rend(); ++it) {
            trial_entry & te = *it;
            if (te.strm != strm_ || te.pos != pos || !te.seq.test(s)) {
                continue;
            }
            seq_node_t * n = te.seq_nodes.find(s);
            if (n != nullptr && !n->node.empty()) {
                *te_out = &te;
                return true;
            }
        }

        // the seq_pos node is keyed by (seq, position), not by the cell: two
        // cells can share the same key. another cell's global entry can own
        // the extracted node, so scan all entries as the last resort.
        for (auto & e : entries_) {
            if (e.pos != pos || !e.seq.test(s)) {
                continue;
            }
            seq_node_t * n = e.seq_nodes.find(s);
            if (n != nullptr && !n->node.empty()) {
                *ge_out = &e;
                return true;
            }
        }

        return false;
    }

    // a mutation that only adds to used/seq_pos (pos_set, seq_add, ext_set)
    void record(uint32_t i, llama_pos pos, const llama_kv_cell_ext & ext, llama_pos shift,
                const std::bitset<LLAMA_MAX_SEQ> & seq) {
        if (i < seen_.size() && (seen_[i] >> 32) == gen_) {
            if (trial_seen(i)) {
                return;
            }
            reserve_trial(1);
            make_trial(i, pos, ext, shift, seq);
            mark_trial(i);
            return;
        }

        reserve_entries(1);

        entry ent;
        ent.i = i;
        ent.pos = pos;
        ent.ext = ext;
        ent.shift = shift;
        ent.seq = seq;

        GGML_ASSERT(entries_.size() < (1ull << 32));
        seen_[i] = (uint64_t(gen_) << 32) | uint32_t(entries_.size());
        entries_.push_back(std::move(ent));
        mark_trial(i);
    }

    void record_rm(uint32_t i, llama_pos pos, const llama_kv_cell_ext & ext, llama_pos shift,
                   const std::bitset<LLAMA_MAX_SEQ> & seq,
                   std::set<uint32_t> & used, std::map<llama_pos, int> (&seq_pos)[LLAMA_MAX_SEQ]) {
        const bool g_seen = i < seen_.size() && (seen_[i] >> 32) == gen_;

        if (g_seen && trial_seen(i)) {
            trial_entry * te = find_trial(current_trial(), i);

            reserve_discard(seq.count() + 1);
            if (te == nullptr) {
                // the trial's first mutation created the global entry, so the
                // global entry is also this trial's snapshot. own the nodes the
                // trial removes next; the global entry keeps the first nodes.
                reserve_trial(1);
                entry * ge = find(i);
                assert(ge != nullptr);
                te = &make_trial(i, ge->pos, ge->ext, ge->shift, ge->seq);
            }
            reserve_trial_nodes(*te, seq.count() + 1);

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) if (seq.test(s)) {
                auto it = seq_pos[s].find(pos);
                assert(it != seq_pos[s].end());
                if (it->second == 1) {
                    auto node = seq_pos[s].extract(it);
                    if (te != nullptr && te->seq.test(s) && !entry_has_seq(*te, s)) {
                        te->seq_nodes.emplace_back(s, std::move(node));
                    } else {
                        discard_seq_.push_back(std::move(node));
                    }
                } else {
                    --it->second;
                }
            }

            auto uit = used.find(i);
            if (uit != used.end()) {
                auto node = used.extract(uit);
                if (te != nullptr && te->pos != -1 && te->used_node.empty()) {
                    te->used_node = std::move(node);
                } else {
                    discard_used_.push_back(std::move(node));
                }
            }
            return;
        }

        if (!g_seen) {
            reserve_entries(1);

            entry ent;
            ent.i = i;
            ent.pos = pos;
            ent.ext = ext;
            ent.shift = shift;
            ent.seq = seq;

            size_t n = 0;
            for_each_seq(seq, [&](int s) {
                auto it = seq_pos[s].find(pos);
                assert(it != seq_pos[s].end());
                if (it->second == 1) ++n;
            });
            ent.seq_nodes.reserve(n);
            ent.used_node = used.extract(i);

            for_each_seq(seq, [&](int s) {
                auto it = seq_pos[s].find(pos);
                assert(it != seq_pos[s].end());
                if (it->second == 1) {
                    ent.seq_nodes.emplace_back(s, seq_pos[s].extract(it));
                } else {
                    --it->second;
                }
            });

            GGML_ASSERT(entries_.size() < (1ull << 32));
            seen_[i] = (uint64_t(gen_) << 32) | uint32_t(entries_.size());
            entries_.push_back(std::move(ent));
            mark_trial(i);
            return;
        }

        // global seen, trial first: own the trial snapshot state and its nodes
        reserve_trial(1);
        trial_entry & te = make_trial(i, pos, ext, shift, seq);
        reserve_trial_nodes(te, seq.count() + 1);

        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) if (seq.test(s)) {
            auto it = seq_pos[s].find(pos);
            assert(it != seq_pos[s].end());
            if (it->second == 1) {
                te.seq_nodes.emplace_back(s, seq_pos[s].extract(it));
            } else {
                --it->second;
            }
        }

        auto uit = used.find(i);
        if (uit != used.end()) {
            te.used_node = used.extract(uit);
        }

        mark_trial(i);
    }

    void record_seq_rm(uint32_t i, llama_pos pos, const llama_kv_cell_ext & ext, llama_pos shift,
                       const std::bitset<LLAMA_MAX_SEQ> & seq, llama_seq_id seq_id,
                       std::set<uint32_t> & used, std::map<llama_pos, int> (&seq_pos)[LLAMA_MAX_SEQ]) {
        const bool last = seq.count() == 1;
        const bool g_seen = i < seen_.size() && (seen_[i] >> 32) == gen_;

        if (g_seen && trial_seen(i)) {
            trial_entry * te = find_trial(current_trial(), i);

            reserve_discard(2);
            if (te == nullptr) {
                // the trial's first mutation created the global entry; use its
                // state as the trial snapshot and own the nodes removed next
                reserve_trial(1);
                entry * ge = find(i);
                assert(ge != nullptr);
                te = &make_trial(i, ge->pos, ge->ext, ge->shift, ge->seq);
            }
            reserve_trial_nodes(*te, 1);

            auto it = seq_pos[seq_id].find(pos);
            assert(it != seq_pos[seq_id].end());
            if (it->second == 1) {
                auto node = seq_pos[seq_id].extract(it);
                if (te != nullptr && te->seq.test(seq_id) && !entry_has_seq(*te, seq_id)) {
                    te->seq_nodes.emplace_back(seq_id, std::move(node));
                } else {
                    discard_seq_.push_back(std::move(node));
                }
            } else {
                --it->second;
            }

            if (last) {
                auto uit = used.find(i);
                if (uit != used.end()) {
                    auto node = used.extract(uit);
                    if (te != nullptr && te->pos != -1 && te->used_node.empty()) {
                        te->used_node = std::move(node);
                    } else {
                        discard_used_.push_back(std::move(node));
                    }
                }
            }
            return;
        }

        if (!g_seen) {
            reserve_entries(1);

            entry ent;
            ent.i = i;
            ent.pos = pos;
            ent.ext = ext;
            ent.shift = shift;
            ent.seq = seq;

            ent.seq_nodes.reserve(1);

            auto it = seq_pos[seq_id].find(pos);
            assert(it != seq_pos[seq_id].end());
            if (it->second == 1) {
                ent.seq_nodes.emplace_back(seq_id, seq_pos[seq_id].extract(it));
            } else {
                --it->second;
            }
            if (last) ent.used_node = used.extract(i);

            GGML_ASSERT(entries_.size() < (1ull << 32));
            seen_[i] = (uint64_t(gen_) << 32) | uint32_t(entries_.size());
            entries_.push_back(std::move(ent));
            mark_trial(i);
            return;
        }

        // global seen, trial first
        reserve_trial(1);
        trial_entry & te = make_trial(i, pos, ext, shift, seq);
        reserve_trial_nodes(te, 1);

        auto it = seq_pos[seq_id].find(pos);
        assert(it != seq_pos[seq_id].end());
        if (it->second == 1) {
            te.seq_nodes.emplace_back(seq_id, seq_pos[seq_id].extract(it));
        } else {
            --it->second;
        }
        if (last) {
            auto uit = used.find(i);
            if (uit != used.end()) {
                te.used_node = used.extract(uit);
            }
        }

        mark_trial(i);
    }

    template <typename Restore>
    void replay(Restore && restore) {
        for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
            restore(*it);
        }
        entries_.clear();
        trial_entries_.clear();
        trial_ranges_.clear();
        discard_used_.clear();
        discard_seq_.clear();
    }

    void clear() {
        entries_.clear();
        trial_entries_.clear();
        trial_ranges_.clear();
        discard_used_.clear();
        discard_seq_.clear();
    }

    size_t size()           const { return entries_.size(); }
    size_t entry_capacity() const { return entries_.capacity(); }
    size_t seen_size()      const { return seen_.size(); }

    size_t bytes() const {
        size_t n = seen_.capacity()*sizeof(uint64_t) + trial_seen_.capacity()*sizeof(uint32_t) +
                   entries_.capacity()*sizeof(entry) +
                   trial_entries_.capacity()*sizeof(trial_entry) +
                   trial_ranges_.capacity()*sizeof(std::pair<size_t, size_t>) +
                   discard_used_.capacity()*sizeof(used_node_t) +
                   discard_seq_.capacity()*sizeof(seq_pos_node_t);
        for (const auto & e : entries_) {
            n += e.seq_nodes.heap_capacity()*sizeof(seq_node_t);
        }
        for (const auto & e : trial_entries_) {
            n += e.seq_nodes.heap_capacity()*sizeof(seq_node_t);
        }
        return n;
    }

private:
    static bool entry_has_seq(const entry & e, llama_seq_id s) {
        return e.seq_nodes.find(s) != nullptr;
    }

    static bool entry_has_seq(const trial_entry & e, llama_seq_id s) {
        return e.seq_nodes.find(s) != nullptr;
    }

    bool trial_seen(uint32_t i) const {
        return i < trial_seen_.size() && trial_seen_[i] == trial_gen_;
    }

    void mark_trial(uint32_t i) {
        if (trial_seen_.size() < seen_.size()) {
            trial_seen_.resize(seen_.size(), 0);
        }
        trial_seen_[i] = trial_gen_;
    }

    // a journal used without begin_trial() (unit tests) still has a range
    trial_view current_trial() {
        if (trial_ranges_.empty()) {
            trial_ranges_.emplace_back(trial_entries_.size(), trial_entries_.size());
        }
        return trial_at(trial_ranges_.size() - 1);
    }

    trial_entry & make_trial(uint32_t i, llama_pos pos, const llama_kv_cell_ext & ext,
                             llama_pos shift, const std::bitset<LLAMA_MAX_SEQ> & seq) {
        if (trial_ranges_.empty()) {
            trial_ranges_.emplace_back(trial_entries_.size(), trial_entries_.size());
        }
        trial_entry te;
        te.i = i;
        te.strm = strm_;
        te.pos = pos;
        te.ext = ext;
        te.shift = shift;
        te.seq = seq;
        trial_entries_.push_back(std::move(te));
        trial_ranges_.back().second = trial_entries_.size();
        return trial_entries_.back();
    }

    void reserve_entries(size_t n) {
        if (entries_.capacity() < entries_.size() + n) {
            entries_.reserve(entries_.capacity()*2 + n + 16);
        }
    }

    void reserve_trial(size_t n) {
        if (trial_entries_.capacity() < trial_entries_.size() + n) {
            trial_entries_.reserve(trial_entries_.capacity()*2 + n + 16);
        }
    }

    void reserve_trial_nodes(trial_entry & te, size_t n) {
        te.seq_nodes.reserve(te.seq_nodes.size() + n);
    }

    void reserve_discard(size_t n) {
        if (discard_used_.capacity() < discard_used_.size() + 1) {
            discard_used_.reserve(discard_used_.size()*2 + 8);
        }
        if (discard_seq_.capacity() < discard_seq_.size() + n) {
            discard_seq_.reserve(discard_seq_.size()*2 + n + 8);
        }
    }

    std::vector<entry> entries_;
    std::vector<trial_entry> trial_entries_;
    std::vector<std::pair<size_t, size_t>> trial_ranges_;
    std::vector<used_node_t> discard_used_;
    std::vector<seq_pos_node_t> discard_seq_;
    std::vector<uint64_t> seen_;
    std::vector<uint32_t> trial_seen_;
    uint32_t strm_ = 0;
    uint32_t gen_ = 0;
    uint32_t trial_gen_ = 0;
};
// meta information about KV cells that can be part of multiple sequences at the same time
// TODO: add unit tests
class llama_kv_cells {
public:
    using seq_set_t = std::bitset<LLAMA_MAX_SEQ>;

    llama_kv_cells() = default;

    // A copy is a data snapshot. It must not inherit the source's journal
    // borrow: the snapshot outlives or aliases a trial, and a mutation of the
    // snapshot would otherwise record into the source's live journal (or a
    // journal that is already gone).
    llama_kv_cells(const llama_kv_cells & other)
        : has_shift(other.has_shift),
          used(other.used),
          pos(other.pos),
          pos_max_(other.pos_max_),
          ext(other.ext),
          shift(other.shift),
          seq(other.seq),
          journal_(nullptr) {
        for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            seq_pos[s] = other.seq_pos[s];
        }
    }

    // Assignment targets are unbound snapshots. A bound object is the live
    // writer of its stream and is only mutated in place: the per-cell journal
    // cannot represent a wholesale replacement (including a size change), so
    // overwriting it would leave a trial that cannot be replayed. The
    // assertion is active in every build; a bound destination fails loudly
    // instead of silently dropping its journal.
    //
    // Copy-and-swap gives the strong exception guarantee, and the destination
    // stays unbound: a copy never inherits the source's journal.
    llama_kv_cells & operator=(const llama_kv_cells & other) {
        if (this != &other) {
            GGML_ASSERT(journal_ == nullptr);
            llama_kv_cells tmp(other);
            swap_data(tmp);
        }
        return *this;
    }

    // A move transfers the borrow to the destination and clears the source,
    // so exactly one live object continues the trial.
    llama_kv_cells(llama_kv_cells && other) noexcept
        : has_shift(other.has_shift),
          used(std::move(other.used)),
          pos(std::move(other.pos)),
          pos_max_(other.pos_max_),
          ext(std::move(other.ext)),
          shift(std::move(other.shift)),
          seq(std::move(other.seq)),
          journal_(other.journal_) {
        for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            seq_pos[s] = std::move(other.seq_pos[s]);
        }
        other.journal_ = nullptr;
    }

    // Move assignment also requires an unbound destination. The destination
    // adopts the source's borrow (the move-construction rule), so the source's
    // trial continues in the moved-to object; the source is cleared and never
    // records again. A bound destination would have to merge two trials, which
    // the journal cannot represent, so it fails loudly.
    llama_kv_cells & operator=(llama_kv_cells && other) noexcept {
        if (this != &other) {
            GGML_ASSERT(journal_ == nullptr);
            journal_ = other.journal_;
            other.journal_ = nullptr;
            swap_data(other);
        }
        return *this;
    }

    void set_journal(cell_journal * journal) {
        journal_ = journal;
    }

    bool is_journal_bound() const {
        return journal_ != nullptr;
    }

    // Host payload capacity of one cells object. This counts vector capacity
    // and container payload entries, not allocator bookkeeping; it excludes
    // the journal (accounted separately by cell_journal::bytes()).
    size_t host_bytes() const {
        size_t n = used.size()*sizeof(uint32_t) +
                   pos.capacity()*sizeof(llama_pos) +
                   ext.capacity()*sizeof(llama_kv_cell_ext) +
                   shift.capacity()*sizeof(llama_pos) +
                   seq.capacity()*sizeof(seq_set_t);
        for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            n += seq_pos[s].size()*(sizeof(llama_pos) + sizeof(int));
        }
        return n;
    }

    // Swap the cell payload without changing either object's address or
    // journal borrow. Used by the shift-backup restore to keep get_cells()
    // references valid across a restore.
    void swap_contents(llama_kv_cells & other) noexcept {
        swap_data(other);
    }

    // Restore one cell from the prepare journal. This is the only writer used
    // by the journal replay; it updates used/seq_pos so the cell invariants
    // hold.
    void restore_cell(uint32_t i, llama_pos p, const llama_kv_cell_ext & e, llama_pos sh,
                      const std::bitset<LLAMA_MAX_SEQ> & s) {
        assert(i < pos.size());

        if (pos[i] != -1) {
            seq_pos_rm(i);
        }

        if (p == -1) {
            used.erase(i);
        } else {
            used.insert(i);
        }

        write_pos(i, p);
        ext[i]   = e;
        seq[i]   = s;
        shift[i] = sh;

        if (p != -1) {
            seq_pos_add(i);
        }
    }

    void reset() {
        for (uint32_t i = 0; i < pos.size(); ++i) {
            write_pos(i, -1);
            ext[i].reset();
            shift[i] =  0;
            seq[i].reset();
        }

        has_shift = false;

        used.clear();

        pos_max_ = -1;

        for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            seq_pos[s].clear();
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

        reset();
    }

    bool is_empty(uint32_t i) const {
        assert(i < pos.size());
        assert((pos[i] < 0 && pos[i] == -1) || pos[i] >= 0);

        return pos[i] == -1;
    }

    // Test/diagnostics: the full used set and seq_pos map, not just counts.
    const std::set<uint32_t> & get_used_set() const { return used; }
    const std::map<llama_pos, int> & get_seq_pos(llama_seq_id s) const { return seq_pos[s]; }

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

    // the position span of the used cells: the largest position present + 1
    // return 0 if no cells are used
    //
    // This is a POSITION span, not a cell-index span. used_max_p1() above
    // answers "how many cell indices are covered"; this answers "how many
    // positions are covered". They differ whenever a cell index is not a
    // position: a unified cache holds two sequences' tokens for the same
    // position in two different cells, so the cell span (2) can exceed the
    // position span (1). The attention width is a position span (see
    // llama_kv_cache::get_n_kv), so it must read this one.
    uint32_t used_pos_span_p1() const {
        uint32_t result = 0;

        for (const uint32_t i : used) {
            const uint32_t p1 = (uint32_t) (pos[i] + 1);

            if (p1 > result) {
                result = p1;
            }
        }

        return result;
    }

    // the index of a cell that holds position p, or -1 when no cell does.
    // The caller shares a hit instead of appending a second cell for a
    // position that already has a slot (see llama_kv_cache::find_slot). A
    // position can still map to more than one cell: a batch may introduce
    // duplicates before apply, and a restored state is not deduplicated.
    //
    // pos_max_ is a conservative upper bound over the positions of used
    // cells. It is monotone between resets: removals and downward shifts may
    // leave it above the real maximum, which only skips the fast path. A
    // position above the bound cannot be present, so the common "new token"
    // miss returns without walking the used set. A hit or an in-range miss
    // keeps the original scan, including its first-cell order when several
    // cells share a position.
    int64_t find_cell_by_pos(llama_pos p) const {
        if (p > pos_max_) {
            return -1;
        }

        for (const uint32_t i : used) {
            if (pos[i] == p) {
                return i;
            }
        }

        return -1;
    }

    bool get_has_shift() const {
        return has_shift;
    }

    void set_has_shift(bool value) {
        has_shift = value;
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

            res.write_pos(j, pos[idx]);
            res.ext[j]   = ext[idx];
            res.seq[j]   = seq[idx];
            res.shift[j] = shift[idx];
        }

        return res;
    }

    // copy the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
    llama_kv_cells cp(const std::vector<uint32_t> & idxs) const {
        llama_kv_cells res;

        res.resize(idxs.size());

        // the per-cell copy carries pos/ext/seq/shift; the stream has_shift
        // flag is not part of this contract (a prepare trial does not change it)
        for (uint32_t j = 0; j < idxs.size(); ++j) {
            const auto idx = idxs[j];

            res.write_pos(j, pos[idx]);
            res.ext[j]   = ext[idx];
            res.seq[j]   = seq[idx];
            res.shift[j] = shift[idx];
        }

        return res;
    }

    // copy the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1]) into out
    // same value set as cp() above, but out keeps its allocated capacity
    void cp_into(const std::vector<uint32_t> & idxs, llama_kv_cells & out) const {
        out.resize(idxs.size());

        for (uint32_t j = 0; j < idxs.size(); ++j) {
            const auto idx = idxs[j];

            out.write_pos(j, pos[idx]);
            out.ext[j]   = ext[idx];
            out.seq[j]   = seq[idx];
            out.shift[j] = shift[idx];
        }

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

            write_pos(idx, other.pos[j]);
            ext[idx]   = other.ext[j];
            seq[idx]   = other.seq[j];
            shift[idx] = other.shift[j];

            if (pos[idx] != -1) {
                seq_pos_add(i + j);
            }
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

            write_pos(idx, other.pos[j]);
            ext[idx]   = other.ext[j];
            seq[idx]   = other.seq[j];
            shift[idx] = other.shift[j];

            if (pos[idx] != -1) {
                seq_pos_add(idx);
            }
        }

    }

    // clear a non-empty cell
    void rm(uint32_t i) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        if (journal_) {
            journal_->record_rm(i, pos[i], ext[i], shift[i], seq[i], used, seq_pos);
        } else {
            seq_pos_rm(i);
            used.erase(i);
        }

        seq[i].reset();

        write_pos(i, -1);
        ext[i].reset();
        shift[i] = 0;
    }

    // note: call only if the cell has seq_id
    // return true if the cell becomes empty
    bool seq_rm(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());
        assert(seq[i].test(seq_id));
        assert(pos[i] != -1);
        assert(seq_id >= 0);

        if (journal_) {
            journal_->record_seq_rm(i, pos[i], ext[i], shift[i], seq[i], seq_id, used, seq_pos);
        } else {
            seq_pos_dec(seq_id, pos[i]);
        }

        seq[i].reset(seq_id);

        if (seq[i].none()) {
            write_pos(i, -1);
            ext[i].reset();
            shift[i] = 0;

            if (!journal_) {
                used.erase(i);
            }

            return true;
        }

        return false;
    }

    // Restore one cell from the prepare journal. The journal owns the nodes
    // removed by the trial, so the inserts here do not allocate. The current
    // cell is compared with the recorded state: only the members the trial
    // added or removed are adjusted.
    bool restore_from_journal(cell_journal & journal, uint32_t i) {
        cell_journal::entry * e = journal.find(i);
        if (e == nullptr) {
            return false;
        }
        return restore_from_journal(journal, *e);
    }

    // The replay already holds the entry. Using it directly avoids a second
    // seen_ lookup per cell and lets the common whole-cell undo reinsert the
    // node handles that this very entry owns.
    bool restore_from_journal(cell_journal & journal, cell_journal::entry & e) {
        const uint32_t i = e.i;
        const seq_set_t cur_seq = seq[i];
        const llama_pos cur_pos = pos[i];
        const bool cur_used = cur_pos != -1;
        const bool old_used = e.pos != -1;

        // remove the entries of the current position. a position change moves
        // every sequence of the cell, so compare positions, not only the bits.
        if (cur_used) {
            for_each_seq(cur_seq, [&](int s) {
                if (cur_pos != e.pos || !e.seq.test(s)) {
                    seq_pos_dec(s, cur_pos);
                }
            });
            if (!old_used) {
                used.erase(i);
            }
        }

        write_pos(i, e.pos);
        ext[i]   = e.ext;
        seq[i]   = e.seq;
        shift[i] = e.shift;

        if (old_used) {
            if (!cur_used) {
                if (!e.used_node.empty()) {
                    used_insert(used, i, std::move(e.used_node));
                } else {
                    cell_journal::entry * src_ge = nullptr;
                    cell_journal::trial_entry * src_te = nullptr;
                    GGML_ASSERT(journal.find_used_source(i, e.pos, &src_ge, &src_te));
                    auto node = src_ge != nullptr ? std::move(src_ge->used_node)
                                                  : std::move(src_te->used_node);
                    used_insert(used, i, std::move(node));
                }
            }
            for_each_seq(e.seq, [&](int s) {
                if (e.pos == cur_pos && cur_seq.test(s)) {
                    // the sequence was present at the same position before and
                    // after the trial, so its map entry never went away
                    return;
                }
                cell_journal::seq_node_t * own = e.seq_nodes.find(s);
                if (own != nullptr && !own->node.empty()) {
                    seq_pos_insert_or_inc(seq_pos[s], e.pos, std::move(own->node));
                    return;
                }
                cell_journal::entry * src_ge = nullptr;
                cell_journal::trial_entry * src_te = nullptr;
                if (journal.find_seq_source(i, e.pos, s, &src_ge, &src_te)) {
                    auto & nodes = src_ge != nullptr ? src_ge->seq_nodes : src_te->seq_nodes;
                    cell_journal::seq_node_t * n = nodes.find(s);
                    GGML_ASSERT(n != nullptr && !n->node.empty());
                    seq_pos_insert_or_inc(seq_pos[s], e.pos, std::move(n->node));
                } else {
                    // the position count was greater than one; the map node
                    // is still present and the iterator increment cannot
                    // allocate (operator[] could)
                    auto pos_it = seq_pos[s].find(e.pos);
                    GGML_ASSERT(pos_it != seq_pos[s].end());
                    ++pos_it->second;
                }
            });
        }

        return true;
    }

    // Restore one cell from the per-trial journal entry. The entry owns the
    // snapshot state and the nodes the trial removed, so this never allocates.
    bool restore_trial(cell_journal & journal, uint32_t i, cell_journal::trial_entry & te) {
        GGML_ASSERT(te.strm == journal.strm());

        const seq_set_t cur_seq = seq[i];
        const llama_pos cur_pos = pos[i];
        const bool cur_used = cur_pos != -1;
        const bool old_used = te.pos != -1;

        if (cur_used) {
            for_each_seq(cur_seq, [&](int s) {
                if (cur_pos != te.pos || !te.seq.test(s)) {
                    seq_pos_dec(s, cur_pos);
                }
            });
            if (!old_used) {
                used.erase(i);
            }
        }

        write_pos(i, te.pos);
        ext[i]   = te.ext;
        seq[i]   = te.seq;
        shift[i] = te.shift;

        if (old_used) {
            if (!cur_used) {
                if (!te.used_node.empty()) {
                    used_insert(used, i, std::move(te.used_node));
                } else {
                    cell_journal::entry * src_ge = nullptr;
                    cell_journal::trial_entry * src_te = nullptr;
                    GGML_ASSERT(journal.find_used_source(i, te.pos, &src_ge, &src_te));
                    auto node = src_ge != nullptr ? std::move(src_ge->used_node)
                                                  : std::move(src_te->used_node);
                    used_insert(used, i, std::move(node));
                }
            }
            for_each_seq(te.seq, [&](int s) {
                if (te.pos == cur_pos && cur_seq.test(s)) {
                    return;
                }
                cell_journal::seq_node_t * own = te.seq_nodes.find(s);
                if (own != nullptr && !own->node.empty()) {
                    seq_pos_insert_or_inc(seq_pos[s], te.pos, std::move(own->node));
                    return;
                }
                cell_journal::entry * src_ge = nullptr;
                cell_journal::trial_entry * src_te = nullptr;
                if (journal.find_seq_source(i, te.pos, s, &src_ge, &src_te)) {
                    auto & nodes = src_ge != nullptr ? src_ge->seq_nodes : src_te->seq_nodes;
                    cell_journal::seq_node_t * n = nodes.find(s);
                    GGML_ASSERT(n != nullptr && !n->node.empty());
                    seq_pos_insert_or_inc(seq_pos[s], te.pos, std::move(n->node));
                } else {
                    auto pos_it = seq_pos[s].find(te.pos);
                    GGML_ASSERT(pos_it != seq_pos[s].end());
                    ++pos_it->second;
                }
            });
        }

        return true;
    }

    // Restore one cell from the per-trial snapshot, whatever its state. The
    // snapshot is the real trial baseline: a later trial can reuse a cell that
    // an earlier trial left used. The journal owns the nodes the trial removed,
    // so this never allocates.
    bool restore_cell_from_snapshot(cell_journal & journal, uint32_t i,
                                    const llama_kv_cells & snap, uint32_t j) {
        const llama_pos old_pos = snap.pos[j];
        const seq_set_t old_seq = snap.seq[j];
        const bool old_used = old_pos != -1;

        const seq_set_t cur_seq = seq[i];
        const llama_pos cur_pos = pos[i];
        const bool cur_used = cur_pos != -1;

        if (cur_used) {
            for_each_seq(cur_seq, [&](int s) {
                if (cur_pos != old_pos || !old_seq.test(s)) {
                    seq_pos_dec(s, cur_pos);
                }
            });
            if (!old_used) {
                used.erase(i);
            }
        }

        write_pos(i, old_pos);
        ext[i]   = snap.ext[j];
        seq[i]   = old_seq;
        shift[i] = snap.shift[j];

        if (old_used) {
            if (!cur_used) {
                cell_journal::entry * src_ge = nullptr;
                cell_journal::trial_entry * src_te = nullptr;
                GGML_ASSERT(journal.find_used_source(i, old_pos, &src_ge, &src_te));
                auto node = src_ge != nullptr ? std::move(src_ge->used_node)
                                              : std::move(src_te->used_node);
                used_insert(used, i, std::move(node));
            }
            for_each_seq(old_seq, [&](int s) {
                if (old_pos == cur_pos && cur_seq.test(s)) {
                    return;
                }
                cell_journal::entry * src_ge = nullptr;
                cell_journal::trial_entry * src_te = nullptr;
                if (journal.find_seq_source(i, old_pos, s, &src_ge, &src_te)) {
                    auto & nodes = src_ge != nullptr ? src_ge->seq_nodes : src_te->seq_nodes;
                    cell_journal::seq_node_t * n = nodes.find(s);
                    GGML_ASSERT(n != nullptr && !n->node.empty());
                    seq_pos_insert_or_inc(seq_pos[s], old_pos, std::move(n->node));
                } else {
                    auto pos_it2 = seq_pos[s].find(old_pos);
                    GGML_ASSERT(pos_it2 != seq_pos[s].end());
                    ++pos_it2->second;
                }
            });
        }

        return true;
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

            write_pos(i, -1);
            ext[i].reset();
            shift[i] = 0;

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
    void seq_add(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());
        assert(pos[i] != -1);
        assert(!seq[i].test(seq_id));

        if (journal_) journal_->record(i, pos[i], ext[i], shift[i], seq[i]);

        // allocate the seq_pos map entry first: if this throws, the cell is
        // unchanged and the journal's old state still matches
        seq_pos_inc(seq_id, pos[i]);
        seq[i].set(seq_id);
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

        if (journal_) journal_->record(i, pos[i], ext[i], shift[i], seq[i]);

        // the used insert can allocate. do it before writing the cell, so a
        // failure leaves the cell empty and matches the journal's old state.
        used.insert(i);

        write_pos(i, p);
    }

    void ext_set(uint32_t i, llama_kv_cell_ext p) {
        assert(i < ext.size());

        if (journal_) journal_->record(i, pos[i], ext[i], shift[i], seq[i]);

        ext[i] = p;
    }

    // pos[i] = pos[i] + d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    bool pos_add(uint32_t i, llama_pos d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        seq_pos_rm(i);

        write_pos(i, pos[i] + d);
        shift[i] += d;

        has_shift = true;

        if (pos[i] < 0) {
            seq[i].reset();
            write_pos(i, -1);
            shift[i] = 0;

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

        write_pos(i, pos[i] / d);
        shift[i] += p_old - pos[i];

        seq_pos_add(i);

        has_shift = true;
    }

private:
    // Every write to pos goes through here so pos_max_ stays an upper bound
    // over the positions of used cells. The bound never decreases except on
    // reset(): a removal or a downward shift leaves it high, which is safe.
    void write_pos(uint32_t i, llama_pos p) {
        pos[i] = p;
        if (p > pos_max_) {
            pos_max_ = p;
        }
    }

    // Swap the cell data without touching the journal borrow. Both assignment
    // operators require an unbound destination, so journal_ is not swapped.
    void swap_data(llama_kv_cells & other) noexcept {
        std::swap(has_shift, other.has_shift);
        used.swap(other.used);
        pos.swap(other.pos);
        std::swap(pos_max_, other.pos_max_);
        ext.swap(other.ext);
        shift.swap(other.shift);
        seq.swap(other.seq);
        for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            seq_pos[s].swap(other.seq_pos[s]);
        }
    }

    bool has_shift = false;

    // set of indices of used cells (i.e. pos[i] != -1, allowed to not have any seq_id)
    std::set<uint32_t> used;

    std::vector<llama_pos> pos;

    // Conservative upper bound over the positions of used cells. write_pos()
    // raises it on every non-negative position write; only reset() lowers it,
    // after used has been cleared.
    llama_pos pos_max_ = -1;

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

    // Active prepare journal; null outside a trial. The mutation methods
    // record the cell's old state before writing it.
    cell_journal * journal_ = nullptr;

    // the set seq_pos[s][p] tells us how many times the position p is currently present for sequence s
    // if the position p is not present, seq_pos[s][p] is not set
    // this way seq_pos[s].begin() and seq_pos[s].rbegin() give us the min/max positions currently in the cache
    //
    // note that we cannot a use an std::set because in some cases a position can occur more than once for the same seq:
    //  - during performing a cache reuse via (rm + add)
    //  - some vision models have input embeddings with repeating positions
    //
    std::map<llama_pos, int> seq_pos[LLAMA_MAX_SEQ];

    // helper functions for updating `seq_pos`, once cell at a time:

    void seq_pos_dec(llama_seq_id s, llama_pos p) {
        auto it = seq_pos[s].find(p);
        GGML_ASSERT(it != seq_pos[s].end());

        if (--it->second == 0) {
            seq_pos[s].erase(it);
        }
    }

    void seq_pos_inc(llama_seq_id s, llama_pos p) {
        seq_pos[s][p]++;
    }

    // Reinsert a node the journal owns. A failed insert means another cell
    // already supplies the same (seq, pos) key: the map stores one node with a
    // count, so increment that count and drop the redundant node. This keeps
    // the restore allocation-free and avoids a separate find before the insert.
    static void seq_pos_insert_or_inc(std::map<llama_pos, int> & m, llama_pos p,
                                      cell_journal::seq_pos_node_t && node) {
        const size_t before = m.size();
        // The replay walks the journal in reverse. Positions usually grow with
        // the cell index, so the new key is often the smallest one; begin() is
        // then the exact hint and the insert is O(1). If the hint is wrong,
        // the standard still inserts correctly, only without the fast path.
        auto hint = (!m.empty() && p < m.begin()->first) ? m.begin() : m.end();
        auto it = m.insert(hint, std::move(node));
        if (m.size() == before) {
            ++it->second;
        }
    }

    static void used_insert(std::set<uint32_t> & s, uint32_t i,
                            cell_journal::used_node_t && node) {
        auto hint = (!s.empty() && i < *s.begin()) ? s.begin() : s.end();
        s.insert(hint, std::move(node));
    }

    // remove cell i
    void seq_pos_rm(uint32_t i) {
        for_each_seq(seq[i], [&](int s) { seq_pos_dec(s, pos[i]); });
    }

    // add cell i
    void seq_pos_add(uint32_t i) {
        for_each_seq(seq[i], [&](int s) { seq_pos_inc(s, pos[i]); });
    }
};

using llama_kv_cells_vec = std::vector<llama_kv_cells>;
