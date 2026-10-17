#pragma once

#include "llama-batch.h"
#include "llama-flashprefill-layout.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"
#include "llama-triattention.h"
#include "llama-xkv-cache.h"
#include "llama-xkv-hot.h"

#include <unordered_map>
#include <vector>
#include <functional>
#include <memory>
#include <mutex>

struct llama_cparams;
struct llama_hparams;

namespace llama_xkv {
class llama_xkv_runtime;
class xkv_graph_snapshot;
class xkv_transaction_coordinator;
struct xkv_state_config;
struct xkv_state_fingerprints;
} // namespace llama_xkv
struct llama_model;
struct llama_context;

// Auto-asymmetric turbo-K upgrade decision (see llama-kv-cache.cpp for the
// full rationale: high-GQA-ratio models amplify turbo K's quantization
// error, so symmetric turbo K+V gets K upgraded to q8_0). Exposed so
// llama-context.cpp's block KV streaming page-geometry pre-scan can size
// its bootstrap allocation off the same effective K type the llama_kv_cache
// constructor will actually use, instead of the raw requested type - the
// two must never diverge or the streaming runtime's bootstrap pool ends up
// sized for the wrong page geometry.
ggml_type llama_kv_cache_resolve_stream_type_k(
        const llama_model & model, const llama_hparams & hparams,
        ggml_type type_k, ggml_type type_v);

// Layer-adaptive per-layer KV precision override (TURBO_LAYER_ADAPTIVE env
// var - see llama-kv-cache.cpp for the mode legend). Exposed, like the
// resolver above, so llama-context.cpp's block KV streaming pre-scan can
// detect ahead of time whether a model will actually get non-uniform
// per-layer KV types: the streamed page pool has one page size and one
// buffer type for the whole arena, so mixed q8_0/turbo2/turbo4 layers would
// otherwise pack differently-shaped rows into pages sized for a layer that
// isn't theirs.
int llama_kv_cache_turbo_layer_adaptive_mode(ggml_type type_v, uint32_t n_layer);

ggml_type llama_kv_cache_turbo_layer_adaptive_type_k(
        int mode, ggml_type type_k, ggml_type type_v, uint32_t il, uint32_t n_layer);
ggml_type llama_kv_cache_turbo_layer_adaptive_type_v(
        int mode, ggml_type type_k, ggml_type type_v, uint32_t il, uint32_t n_layer);

//
// llama_kv_cache
//

class llama_kv_cache_context;

class llama_kv_cache : public llama_memory_i {
    friend class llama_xkv::llama_xkv_runtime;
public:
    struct rerot_resolved_cell {
        uint32_t stream = 0;
        uint32_t cell = 0;
        llama_pos storage_pos = 0;
        llama_pos virtual_pos = 0;
        llama_kv_rerot_meta meta;
    };

    struct rerot_resolved_view {
        uint64_t episode_id = 0;
        llama_rerot_node_id reader = LLAMA_REROT_NODE_INVALID;
        llama_pos query_virtual_pos = 0;
        std::vector<rerot_resolved_cell> cells;
    };

    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        // Physical hot slot indices (for bounded hot cache; empty when hot == logical)
        std::vector<idx_vec_t>    hot_idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    // Overwrite victims collected atomically before any mutation. The hot-only
    // Overwrite victim snapshot: records one overwritten cell so pool/store/
    // cells can be rolled back exactly when apply or graph compute fails.
    // Victim pool/store entries are released only at postcompute_success, so
    // a failure path never needs to re-bind pool rows or resurrect tensor data.
    struct xkv_victim_snapshot {
        uint32_t stream = 0;
        uint32_t cell   = 0;
        llama_pos pos = -1;
        llama_kv_cell_ext ext;
        llama_pos shift = 0;
        // Exact full multi-ref membership. Overwriting a shared cell would
        // destroy co-refs (one cell holds one content), so shared victims are
        // rejected at collect time; the full set is still snapshotted so any
        // restore is bit-identical.
        llama_kv_cells::seq_snapshot_t seqs;
        llama_kv_rerot_meta rerot;
        uint64_t pid = 0;
        uint64_t gen = 0;
        // Presence at collect time. Pool release observes only in_pool rows
        // (aligned 1:1 hot-only triple); store removal observes only in_store
        // rows so factored payloads never reach the pool and are never missed.
        bool in_pool = false;
        uint32_t hot_row = 0;
        bool in_store = false;
        llama_xkv::xkv_state store_state = llama_xkv::xkv_state::hot_writing;
    };
    struct xkv_overwrite_victims {
        std::vector<xkv_victim_snapshot> items;
        bool empty() const { return items.empty(); }
    };

    // One applied token awaiting postcompute confirmation: cell location plus
    // the payload identity and hot row needed for commit or full rollback.
    struct xkv_applied_entry {
        // (stream, idx) pairs in the hook args below reuse these two fields.
        uint32_t stream   = 0;
        uint32_t cell     = 0;
        uint64_t pid      = 0;
        uint64_t gen      = 0;
        uint32_t hot_slot = 0;
        bool     has_hot  = false;
    };

    // TODO: refactor the memory instances to not depend on `llama_model`
    //       instead pass all necessary info (e.g. hparams, dev layers, arch, etc.) directly
    //       likely through `struct llama_memory_params`
    llama_kv_cache(
            const llama_model & model,
          const llama_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
               llama_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share,
        const llama_cparams   * cparams,
                 const char *   name_tag = "",
                          size_t kv_stream_stage_bytes,
                          void * kv_stream_phase_arena,
                          size_t kv_stream_maximum_pool_bytes)
    ~llama_kv_cache();

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    uint32_t get_kv_capacity() const override;
    uint32_t get_kv_used()     const override;
    uint32_t get_kv_seq_used(llama_seq_id seq_id) const override;

    // XKV-owned admission + safe-boundary maintenance overrides. Admission
    // fills the authoritative public snapshot via the runtime coordinator
    // (factor/workspace planner bounds included); OFF maps legacy exactly.
    // Maintenance runs one forced domain-partitioned seal attempt and maps
    // the outcome to PROGRESS / NO_ACTION / RETRY_STALE / ERROR.
    bool get_admission_snapshot(struct llama_memory_admission_snapshot * out) const override;
    llama_memory_maintenance_status maintain_safe_boundary() override;
    // Observed XKV runtime/store snapshot. False (zeroed *out) when no
    // runtime is bound: the caller reports empty/not_evaluated.
    bool get_xkv_runtime_snapshot(struct llama_memory_xkv_runtime_snapshot * out) const override;


    void clear(bool data) override;

    // Failure-reporting clear: preflight all, commit store then pool, reset
    // Failure-reporting clear: preflight all, commit pool then store, reset
    // cells last. Never throws; false leaves every subsystem bit-identical
    // (pool snapshot insurance + store rechecked-preflight contract + cells
    // last; see impl).
    bool try_clear(bool data, std::string * err = nullptr) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool has_kv_stream_targets() const override;
    std::vector<llama_kv_stream_target> get_kv_stream_targets() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    llama_memory_kv_reclaim_result reclaim_kv(const llama_memory_kv_reclaim_request & request) override;

    bool positions_are_sparse() const override;

    // state_read, plus the cells the restored tokens were placed in
    void state_read_sinfo(
            llama_io_read_i & io,
               llama_seq_id   seq_id,
      llama_state_seq_flags   flags,
          slot_info_vec_t *   sinfos_out,
    const slot_info_vec_t *   sinfos_in);

    //
    // llama_kv_cache specific API
    //

    void init_triattention(const char * stats_path, double ratio, uint32_t recent_window, const llama_cparams & cparams);

    uint32_t get_size()     const;
    uint32_t get_n_stream() const;
    std::vector<uint32_t> get_layer_ids() const;
    ggml_tensor * get_k_storage(int32_t il) const;
    ggml_tensor * get_v_storage(int32_t il) const;
    bool get_v_transposed() const;

    bool kv_stream_adapt(uint32_t active_tokens, uint32_t query_tokens);
    bool kv_stream_resize_pool(
        size_t pool_bytes, uint32_t active_tokens, uint32_t ring_slots);

    bool get_has_shift() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

        const llama_cparams   * cparams,
                 const char *   name_tag = "",
                          size_t kv_stream_stage_bytes,
                          void * kv_stream_phase_arena,
                          size_t kv_stream_maximum_pool_bytes)
    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;

    // active cell count when position p lives in physical cell p; 0 for any non-contiguous layout
    uint32_t get_n_kv_pos_contiguous(const slot_info & sinfo, const llama_ubatch & ubatch) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    // Bounded-hot physical views spanning the full hot tensor rows
    // (hot_kv_size per stream). The XKV attention path gathers indexed hot
    // rows from these; logical n_kv may exceed the hot tensor, so the graph
    // must never size physical views from logical counts. Stock get_k/get_v
    // semantics are untouched. Returns nullptr (logged) when unbounded.
    ggml_tensor * get_hot_k(ggml_context * ctx, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * get_hot_v(ggml_context * ctx, int32_t il, const slot_info & sinfo) const;

    // TurboQuant: get rotation matrices (stored as row-major C arrays)
    // turbo_rotation = R (forward rotation, for Q pre-rotate-queries)
    // turbo_rotation_inv = R^T = R^{-1} (inverse rotation, for V output un-rotation)
    ggml_tensor * get_turbo_rotation() const { return turbo_rotation; }
    ggml_tensor * get_turbo_rotation_inv() const { return turbo_rotation_inv; }

    // TurboQuant InnerQ: per-channel scale_inv for Q/V equalization
    ggml_tensor * get_turbo_innerq_scale_inv() const { return turbo_innerq_scale_inv; }

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // Compact KV cache: pack all used cells to [0, used) for each stream
    // Moves K/V tensor rows and updates cell metadata atomically
    // Must be called after llama_synchronize() and before next init_batch()
    void compact();
    // Used by update after synchronization. Turbo K is decoded, phase-shifted
    // and requantized; WHT alone never makes a position shift a no-op.
    void shift_turbo_keys(const llama_cparams & cparams);

    // For tests / diagnostic inspection of the shift compute graph
    ggml_cgraph * build_graph_shift_for_testing(llm_graph_result * res, llama_context * lctx) const {
        return build_graph_shift(res, lctx);
    }

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    // When victims_out != nullptr and dry_run == false, overwrite victims are
    // collected (preflighted, zero mutation on rejection), new cells are bound
    // in the store, and victim release is deferred to the caller so pool commit
    // and victim release stay ordered after a successful commit.
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch, bool dry_run = false,
                        xkv_overwrite_victims * victims_out = nullptr);

    // Reserve real hot slots for prepared sinfos and fill sinfo.hot_idxs.
    // Shared by init_batch and the iSWA/hybrid wrappers so every bounded-hot
    // write path uses identical reservation semantics. On success, reservations
    // are moved to out_reservations aligned 1:1 with sinfos/ubatches; on
    // failure nothing is returned and prior partial reservations roll back.
    bool reserve_hot_slots(slot_info_vec_t & sinfos,
                            const std::vector<llama_ubatch> & ubatches,
                            std::vector<llama_xkv::xkv_hot_reservation> & out_reservations,
                            std::string * err = nullptr);

    // Positions-aware landmark rebuild hook for bounded-SR removals. The
    // store refuses remove/pack without a semantic rebuild; the runtime owns
    // the rebuild and the cache must ask BEFORE cells mutate. Args are
    // (stream, idx) pairs about to be released. Return false to fail closed
    // with zero mutation. Unset (nullptr) preserves legacy behavior: proceed
    // to the store call and propagate its verdict. Runs once per collection
    // (overwrite victims, removal sets, rollback entry sets); the deferred
    // success-path release does not re-ask for already-collected victims.
    // Seal publish paths never invoke this (sealing owns its landmarks).
    using xkv_removal_cells = std::vector<std::pair<uint32_t, uint32_t>>;
    using xkv_removal_rebuild_fn =
        std::function<bool(const xkv_removal_cells & cells, std::string * err)>;
    void set_removal_rebuild_hook(xkv_removal_rebuild_fn fn);

    // Read-only overwrite-victim collection + cross-subsystem preflight.
    // Returns false (zero mutation) when any victim fails pool/store validation.
    bool collect_overwrite_victims(const slot_info & sinfo, xkv_overwrite_victims & out,
                                    std::string * err = nullptr);

    // Release already-preflighted victims at postcompute_success: pool first,
    // then store. Re-validated at release; failures surface as fail-closed
    // errors without touching cells.
    bool release_collected_victims(const xkv_overwrite_victims & victims, std::string * err = nullptr);

    void verify_bound_hot_rows(const std::vector<xkv_applied_entry> & entries);

    // Restore victim cells from snapshots (pool/store victim entries are still
    // bound on every path that calls this). Used by apply/store-register
    // failure and by postcompute_failure before the next decode.
    void restore_victim_cells(const xkv_overwrite_victims & victims);

    // Remove new store entries and drop new cells for applied entries
    // [base, size). Victim pool/store entries are untouched. No-op on view
    // caches, which never own cell lifecycle. Gated (empty pool triple) with
    // bit-identical false; the caller restores victims and the reservation
    // only after true.
    bool rollback_applied_entries(const std::vector<xkv_applied_entry> & entries, size_t base,
                                  std::string * err = nullptr);

    // One coordinated rollback transaction for postcompute_failure: store
    // removal of new payloads, then pool release of new rows, then new-cell
    // drops, then victim-cell restores. Best-effort across steps with a
    // combined status: every step is attempted, and false is returned (never
    // a silent log-and-continue) when any fallible step fails. No-op on view
    // caches. Victim pool/store entries are never released here.
    bool rollback_applied_batch(const std::vector<xkv_applied_entry> & entries,
                                  const std::vector<xkv_overwrite_victims> & victims,
                                  std::string * err = nullptr);

    // Aligned removal release: the hot-only triple addresses exactly the
    // tracked payloads bound in the pool; the store lists cover every tracked
    // payload including factored rows. Shared payload IDs are deduplicated so
    // last-ref release frees once and earlier shared drops release nothing.
    struct xkv_removal_release {
        std::vector<uint64_t> store_pids;
        std::vector<uint64_t> store_gens;
        std::vector<uint64_t> hot_pids;
        std::vector<uint64_t> hot_gens;
        std::vector<uint32_t> hot_slots;
        bool empty() const { return store_pids.empty() && hot_pids.empty(); }
    };

    // Read-only collection + cross-subsystem preflight for cell deletions.
    // idxs are logical cell indices in the given stream's cells. Untracked
    // residue (pid 0 or absent from both subsystems) is skipped freely.
    // Returns false with zero mutation on any validation failure.
    bool collect_removal_release(uint32_t stream, const std::vector<uint32_t> & idxs,
                                  xkv_removal_release & out, std::string * err = nullptr);

    // Execute a preflighted removal: store first, then pool. On pool failure
    // the store bindings are re-registered so no subsystem stays mutated.
    // Cells are mutated only by the caller after success.
    bool execute_removal_release(const xkv_removal_release & rel, std::string * err = nullptr);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;


    // I64 shifted hot-row indices for Turbo write-back (see k_shift_rows).
    void set_input_k_shift_rows(ggml_tensor * dst) const;
    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    // inkling: fill dst I32 [n_kv, n_tokens] with flat rel-bias gather indices,
    // idx(i, j) = i*(extent + 1) + rel; empty/out-of-band cells map to the zero-bias column `extent`
    void set_input_pos_rel_flat(ggml_tensor * dst, const llama_ubatch * ubatch, uint32_t extent) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

    // Test-only accessor: the persistent world (differential probes).
    const llama_rerot_shared_world & get_rerot_world_for_test() const { return rerot_world; }
    const std::unordered_map<llama_seq_id, std::vector<uint64_t>> & get_rerot_world_owned_for_test() const { return rerot_world_owned; }

private:
    bool compact_planned(llama_kv_cells_vec & planned_cells);
    const llama_model & model;
    const llama_hparams & hparams;

    // Runtime-owned positions-aware landmark rebuild (bounded SR). See
    // xkv_removal_rebuild_fn docs; null means legacy store-verdict behavior.
    xkv_removal_rebuild_fn removal_rebuild;

    // Invoke the rebuild hook when set; true when unset. Read-only.
    bool run_removal_rebuild_hook(const xkv_removal_cells & cells, std::string * err);

    // Shared precommit body; probe_store=false makes zero store calls.
    // Private: external callers use precommit_hot_release / _nostore.
    bool precommit_hot_release_impl(
        const llama_xkv::xkv_hot_release_plan & plan,
        bool probe_store,
        std::string * err);

    // Bounded context-shift transaction (update() with do_shift on a bounded
    // cache). Prebuilds phase-aware SR landmarks via store COW packs before
    // committing hot rotation + cell positions + stamps, all under
    // coordinator exclusion. Any preflight/build failure returns false with
    // cells keeping advanced positions + pending deltas (the stock
    // retryable protocol) and store/pool untouched. Private: called from
    // update() only.
    bool update_shift_bounded(llama_context * lctx, std::string * err);

    // Maintenance-exclusion id source for the shift transaction.
    uint64_t xkv_shift_maint_id_ = 1;

    // Build the store-held removal gate callback from a precomputed aligned
    // (pid, gen, slot) triple. Pool-only with zero store calls, so it is
    // safe under the store lock: verifies every triple pid is in the store's
    // removal set with matching gen/slot, then validates + releases inside
    // one pool lock (atomic all-or-nothing). False on any mismatch; an empty
    // triple (or null pool) is a vacuous true. Never throws.
    llama_xkv::xkv_removal_precommit_fn make_pool_removal_gate(
        const std::vector<uint64_t> & pids,
        const std::vector<uint64_t> & gens,
        const std::vector<uint32_t> & slots) const;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
    };

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // True when constructed unified. Bounded-hot XKV is rejected without it.
    const bool kv_unified = true;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    // env: LLAMA_ATTN_ROT_DISABLE
    bool attn_rot_k = false;
    uint32_t attn_rot_k_nrot = 0;
    bool attn_rot_v = false;

    // if all layers participating in the cache have constant head size, the value is stored here
    // otherwise the value is -1
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    // pre-computed hadamard martrices
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // FlashPrefill owner mutation epoch (see flashprefill_epoch()). Bumped by
    // fp_bump() on every metadata mutation listed on the accessor, but ONLY
    // once the flash path has activated (fp_active): OFF keeps zero
    // per-mutation overhead — a single predictable branch, no counter write,
    // no allocation, no scheduling effect. Starts at 1 so a zero epoch
    // unambiguously means "never stamped". Activation happens exclusively via
    // flashprefill_enable_tracking(), which the flash build calls first and
    // captures stamps afterwards, so no pre-activation state can validate.
    uint64_t fp_epoch = 1;
    mutable bool fp_active = false;

    void fp_bump() {
        if (fp_active) {
            ++fp_epoch;
        }
    }

    // Persistent RERoT shared world (2026-09-22 twelfth round, fourth
    // question: most frontiers are numeric-only). The structural pass of
    // the multi-reader layout builder lives in llama_rerot_shared_world and
    // is maintained INCREMENTALLY by the tracked mutation entry points:
    //   - apply_ubatch (non-dry): upsert of every written cell (ring
    //     recycle = same key index, new content);
    //   - rerot_publish_run / rerot_reclassify_run: set_key_meta batch;
    //   - rerot_add_run_ref: ownership only (no record change needed —
    //     ownership is per-reader, rebuilt per layout from seq_has).
    // Every OTHER mutation path is covered by the CellGeneration counter
    // (bumped by every cells mutation): the world records the generation it
    // was built at, and any mismatch forces one full rebuild — the same
    // lazy opt-in pattern as fp_epoch (set_generation_enabled is called
    // once when the world first activates; OFF keeps zero overhead).
    mutable llama_rerot_shared_world rerot_world;
    mutable bool rerot_world_active = false;
    mutable bool rerot_world_valid = false;
    mutable uint64_t rerot_world_gen = 0;   // cells generation at last build
    // Pending increments for a not-yet-valid world (mutations that happen
    // while the world is invalid are covered by the full rebuild).
    mutable std::vector<llama_rerot_key_record> rerot_world_pending;
    // Ownership columns (thirteenth round): per-sequence bitsets indexed by
    // the world's RECORDS POSITION (same index space as the layout's
    // ownership bitmaps — run rows and the untagged order). Maintained by
    // the same tracked entry points as the world (apply purge clears the
    // victim's bits, apply upsert rewrites the written cell's bits, add_run_ref
    // sets the referenced cell's bit); a world rebuild refills them in the
    // same scan. Lazy per-sequence columns: only sequences that actually
    // own cells get a column. A missing column reads as all-zero (the
    // sequence owns nothing) — the layout's ownership probe.
    mutable std::unordered_map<llama_seq_id, std::vector<uint64_t>> rerot_world_owned;

    // Layout assembly + builder-output scratch (fourteenth round): the
    // per-frontier build reuses these buffers across frontiers (capacity
    // and warmed pages survive), eliminating the steady-state
    // allocations: rerot_assembly_scratch feeds the final layout vectors,
    // rerot_multi_scratch feeds the multi-reader builder's per-reader
    // outputs. Not serialized; mutable because the build path is const.
    mutable llama_rerot_attn_layout rerot_assembly_scratch;
    mutable std::vector<std::vector<llama_rerot_query_layout>> rerot_multi_scratch;
    mutable std::vector<llama_pos> rerot_qvp_sink;

    friend class llama_kv_cache_context;

    // Feed the world: called by the tracked mutation entry points. When the
    // world is valid, the increment is applied immediately; when invalid,
    // the records are dropped (the rebuild rescans the cells).
    void rerot_world_upsert(const std::vector<llama_rerot_key_record> & recs) const {
        if (!rerot_world_valid || recs.empty()) {
            return;
        }
        rerot_world.upsert_keys(recs);
    }

    // Meta-only refresh (publish/reclassify): same discipline.
    void rerot_world_set_meta(const std::vector<llama_rerot_key_record> & recs) const {
        if (!rerot_world_valid || recs.empty()) {
            return;
        }
        try {
            rerot_world.set_key_meta(recs);
        } catch (...) {
            // A meta update escaping its run bucket (should not happen on
            // the tracked paths): fail safe, drop the world.
            rerot_world_valid = false;
        }
    }

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    struct kv_stream_runtime_owner {
        using feedback_fn_t = bool (*)(
            void *, uint64_t *, uint64_t *, double *, uint32_t *,
            uint32_t *, uint32_t *, uint32_t *);
        using span_feedback_fn_t = bool (*)(void *, double);
        using reconfigure_fn_t = bool (*)(void *, uint32_t, uint32_t);
        using repartition_fn_t = bool (*)(void *, uint32_t);
        using decode_layout_fn_t = bool (*)(void *, uint32_t);
        using mark_dirty_rows_fn_t = bool (*)(void *, const int64_t *, size_t);
        using resize_pool_fn_t = bool (*)(void *, size_t, uint32_t, uint32_t);

        void * runtime = nullptr;
        void (*free_fn)(void *) = nullptr;
        feedback_fn_t feedback_fn = nullptr;
        span_feedback_fn_t span_feedback_fn = nullptr;
        reconfigure_fn_t reconfigure_fn = nullptr;
        repartition_fn_t repartition_fn = nullptr;
        decode_layout_fn_t decode_layout_fn = nullptr;
        mark_dirty_rows_fn_t mark_dirty_rows_fn = nullptr;
        resize_pool_fn_t resize_pool_fn = nullptr;
        uint32_t layer_count = 0;
        uint32_t minimum_ring_slots = 0;
        uint32_t decode_layout_pages = 0;
        uint32_t starved_evaluations = 0;
        uint32_t overprovisioned_evaluations = 0;
        uint32_t evaluations_since_repartition = UINT32_MAX;
        uint64_t previous_deadline_samples = 0;
        uint64_t previous_deadline_misses = 0;
        int64_t previous_adapt_us = 0;
        uint32_t previous_query_tokens = UINT32_MAX;

        ~kv_stream_runtime_owner() {
            if (runtime != nullptr) {
                free_fn(runtime);
            }
        }
    };

    // Declared before ctxs_bufs so the custom buffers release their runtime
    // references before this owner releases the initial reference.
    kv_stream_runtime_owner kv_stream_runtime;
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    // TODO: temporary until we refactor to be able to share the same cells between 2 kv caches [TAG_KV_CACHE_SHARE_CELLS]
    llama_kv_cache * other;

    std::shared_ptr<llama_kv_cells_vec> v_cells_impl;

    llama_kv_cells_vec & v_cells;

    // XKV shared store handle (nullable when XKV OFF; shared across other and native layer aliases)
    std::shared_ptr<llama_xkv::llama_xkv_cache_store> xkv_store;

    // Physical hot slot pool for XKV dense/SR modes (nullptr for OFF and SHADOW)
    std::shared_ptr<llama_xkv::xkv_hot_slot_pool> xkv_hot_pool;
    // Cache-owned XKV runtime coordinator + single transaction coordinator
    // gating seal/pack/MTP/fence readers. Both null for OFF/MTP/rejected.
    // Shared (shared_ptr) across every cache sharing one store: a per-cache
    // coordinator/runtime pair could race maintenance against the same
    // store. Maintenance itself is serialized by the runtime mutex; the
    // MTP/draft alias never receives a runtime (init returns early), so its
    // layer gate and maintenance stay false and its graph can neither route
    // XKV (no runtime) nor stock attention on truncated tensors
    // (can_use_legacy_attention is false while bounded: fail closed).
    std::shared_ptr<llama_xkv::llama_xkv_runtime> xkv_runtime;
    std::shared_ptr<llama_xkv::xkv_transaction_coordinator> xkv_tx_coord;

    // One executor per physical backend device. Created only for an enabled
    // XKV store, then shared by native sealing, Tri scoring, state I/O, and
    // every device-owned segment so backend setup is never repeated per row
    // or segment.
    mutable std::mutex xkv_executor_mutex;
    mutable std::unordered_map<ggml_backend_dev_t, std::shared_ptr<struct ggml_backend>> xkv_executors;

    uint32_t hot_kv_size = 0;
    bool xkv_dense_sr = false;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // Current per-sequence write classification. These are control-plane
    // values, not physical-cell metadata, and are reset on cache clear.
    std::vector<llama_kv_rerot_meta> rerot_write_tags;
    std::vector<llama_rerot_reader_state> rerot_reader_views;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    // TriAttention importance scorer and configuration
    std::unique_ptr<triattention_scorer> tri_scorer;
    double tri_ratio = 3.0 / 32.0;
    uint32_t tri_recent_window = 128;

    // TurboQuant rotation matrices (128x128, row-major stored)
    ggml_tensor * turbo_rotation = nullptr;      // R (forward rotation)
    ggml_tensor * turbo_rotation_inv = nullptr;   // R^T = R^{-1} (inverse rotation)

    // TurboQuant InnerQ: per-channel scale_inv for Q/V equalization (128 floats)
    ggml_tensor * turbo_innerq_scale_inv = nullptr;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * rot,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale,
                       uint32_t   il) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    // XKV code-stream trailer (PR11, owned by XkvStateCodec): factored code
    // streams appended after legacy bytes on full-state saves and on per-seq
    // saves carrying factored rows (RAM demote/restore, prompt cache, slot
    // files); validated and imported on restores with cell rebind. OFF and
    // PARTIAL_ONLY checkpoints write zero extra bytes.
    void xkv_state_write_trailer(llama_io_write_i & io, llama_seq_id seq_id,
                                  llama_state_seq_flags flags) const;
    void xkv_state_read_trailer(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags,
                                const slot_info * granted, uint32_t granted_strm);
    llama_xkv::xkv_state_fingerprints xkv_state_expected_fps(
        const llama_xkv::xkv_state_config & cfg) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1);
    // sinfo_in, when set, replaces the find_slot call: the cells are given by the caller
    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1, const slot_info * sinfo_in = nullptr);    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);
};

class llama_kv_cache_context : public llama_memory_context_i {
    friend class llama_xkv::llama_xkv_runtime;
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches,
            std::vector<llama_xkv::xkv_hot_reservation> hot_reservations = {});

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    // Success commits store rows and releases deferred victims atomically
    // (victims release only after a successful commit); failure runs one
    // coordinated rollback transaction (store -> pool -> cells -> victims).
    // Both are exactly-once; repeat calls return the recorded outcome.
    bool postcompute_success() override;
    bool postcompute_failure() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    bool has_kv_stream_targets() const override;
    std::vector<llama_kv_stream_active_target> get_kv_stream_active_targets() const override;

    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;
    uint32_t get_n_kv_pos_contiguous() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    ggml_type layer_type_k(int32_t il) const;
    ggml_type layer_type_v(int32_t il) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // Bounded-hot physical views for the current ubatch (see cache-level
    // Graph-facing bounded-hot physical views for the current ubatch (exact
    // spelling required by the XKV attention path; see cache-level
    // get_hot_k/get_hot_v). Same call hazards as get_k/get_v.
    ggml_tensor * get_xkv_hot_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_xkv_hot_v(ggml_context * ctx, int32_t il) const;

    // Per-layer XKV gate for graph routing: true only for target-trunk
    // owning layers. False for SWA/recurrent/MTP and when no runtime exists.
    bool xkv_layer_enabled(uint32_t il) const;

    // Per-KV-head snapshot covering all visible semantic cells (history +
    // current). Delegates to the cache runtime; payload/positions stay
    // cells-owned. Exact spelling required by the XKV attention path.
    bool build_xkv_graph_snapshot(
        uint32_t il,
        uint32_t kv_head,
        float kq_scale,
        float logit_softcap,
        std::unique_ptr<llama_xkv::xkv_graph_snapshot> & out,
        std::string * err = nullptr) const;

    // Minimal read accessors for the runtime snapshot builder (no mutation).
    llama_kv_cache * get_kv() const;
    size_t get_ubatch_index() const;
    size_t get_ubatch_count() const;
    size_t get_sinfo_count() const;
    const llama_ubatch & get_ubatch_at(size_t i) const;
    const llama_kv_cache::slot_info & get_sinfo_at(size_t i) const;


    // TurboQuant rotation accessors
    ggml_tensor * get_turbo_rotation() const;
    ggml_tensor * get_turbo_rotation_inv() const;

    // Override virtual methods from llama_memory_context_i
    ggml_tensor * get_turbo_rot_forward() const override;
    ggml_tensor * get_turbo_rot_inverse() const override;

    // TurboQuant InnerQ: per-channel scale_inv for Q/V equalization
    ggml_tensor * get_turbo_innerq_scale_inv() const override;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be laid out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    bool rerot_active() const;
    const llama_rerot_attn_layout & get_rerot_attn_layout() const;

    //
    // FlashPrefill legal-fragment planning (CacheFragments)
    //
    // Explicit build/get accessor keyed by the OWNED ubatch (index into this
    // context's ubatch vector; the ubatch storage is owned here via shared
    // data, so no borrowed-pointer lifetime hazard) plus role and capacities.
    // The role is a frozen llama_flashprefill_role value supplied by the
    // caller (ContextIntegration descriptor when available); capacities come
    // from params. Existing memory-context wrappers call build directly from
    // the graph attention KV input; groups() output feeds raw-Q RoPE.
    //
    // build returns true iff the layout is eligible and fresh. False means
    // either policy-ineligible (reason in get_layout().dense_reason, no error
    // text) or a hard error (get_layout().error set, error out-param set
    // when provided). get returns the last built layout (cleared/ineligible
    // when nothing was built). is_fresh revalidates owner stamps without
    // rebuilding. Layouts never outlive mutation: next()/apply() drop them
    // (conservative per-graph rebuild) and every owner mutation advances the
    // stamps they are validated against.
    bool flashprefill_build_layout(
        uint32_t ubatch_index,
        int32_t role,
        const llama_flashprefill_layout_params & params,
        std::string * error = nullptr) const;

    // Graph-attention entry point: builds for the CURRENT owned ubatch
    // (i_cur), since graph KV inputs only see get_ubatch() and never the
    // index. Identical keying/freshness semantics to the indexed overload;
    // fails closed (false + reason) on full/update contexts without ubatches.
    bool flashprefill_build_current_layout(
        int32_t role,
        const llama_flashprefill_layout_params & params,
        std::string * error = nullptr) const;

    const llama_flashprefill_layout & flashprefill_get_layout() const;

    bool flashprefill_layout_is_fresh() const;

    const llama_flashprefill_build_key & flashprefill_layout_key() const;

    ggml_tensor * build_input_rerot_q_indices(ggml_context * ctx) const;
    ggml_tensor * build_input_rerot_q_pos(ggml_context * ctx, uint32_t n_pos) const;
    ggml_tensor * build_input_rerot_entries(ggml_context * ctx) const;
    ggml_tensor * build_input_rerot_offsets(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_rerot_q_indices(ggml_tensor * dst) const;
    void set_input_rerot_q_pos(ggml_tensor * dst, uint32_t n_pos) const;
    void set_input_rerot_entries(ggml_tensor * dst) const;
    void set_input_rerot_offsets(ggml_tensor * dst) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_pos_rel_flat(ggml_tensor * dst, const llama_ubatch * ubatch, uint32_t extent) const; // inkling

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;

private:
    llama_memory_status status;

    llama_kv_cache * kv = nullptr;
    llama_context * lctx = nullptr;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    // Move-only RAII reservations corresponding to sinfos
    std::vector<llama_xkv::xkv_hot_reservation> hot_reservations;

    // Applied entries awaiting postcompute confirmation. Records every applied
    // Applied entries awaiting postcompute confirmation. A compute failure
    // rolls back pool + store + cells atomically from these records.
    std::vector<llama_kv_cache::xkv_applied_entry> applied_entries;
    // Deferred overwrite victims, released only after a successful pool commit.
    std::vector<llama_kv_cache::xkv_overwrite_victims> pending_victims;
    bool postcompute_finalized = false;
    bool postcompute_ok = false;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;

    mutable bool rerot_layout_ready = false;
    mutable llama_rerot_attn_layout rerot_layout;
    // Assembly scratch (fourteenth round): the per-frontier layout is built
    // into a FRESH object every call, so entries/groups/query_offsets
    // reallocate from zero each time (the reserve pass still pays one
    // allocation + zeroing per vector). The scratch keeps the vectors alive
    // across frontiers: build into it, then move it into rerot_layout —
    // capacity survives the move and the next frontier reuses it.

    // FlashPrefill planning cache (mutable so const graph KV inputs can build;
    // same discipline as rerot_layout above). Keyed by owned ubatch + role +
    // capacities; validated against owner stamps on every use.
    mutable bool fp_key_valid = false;
    mutable llama_flashprefill_build_key fp_key;
    mutable llama_flashprefill_layout fp_layout;
    mutable uint64_t fp_cells_epoch = 0;
    mutable std::vector<llama_flashprefill_cell_stamp> fp_cell_stamps;
};
