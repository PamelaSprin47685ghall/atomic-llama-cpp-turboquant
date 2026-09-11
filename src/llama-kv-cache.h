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

//
// llama_kv_cache
//

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
        const llama_cparams   * cparams = nullptr);

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

    bool get_has_shift() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    // Actual storage types by logical model layer (including boundary-V
    // overrides and reused layers), without allocating tensor views.
    ggml_type layer_type_k(int32_t il) const;
    ggml_type layer_type_v(int32_t il) const;

    std::vector<uint32_t> get_layer_ids() const;
    ggml_tensor * get_k_storage(int32_t il) const;
    ggml_tensor * get_v_storage(int32_t il) const;

    const llama_model   & get_model()   const { return model; }
    const llama_hparams & get_hparams() const { return hparams; }
    bool                  get_v_trans() const { return v_trans; }
    bool                  get_attn_rot_k() const { return attn_rot_k; }
    bool                  get_attn_rot_v() const { return attn_rot_v; }
    const std::unordered_map<int64_t, std::vector<float>> & get_attn_rot_hadamard() const { return attn_rot_hadamard; }
    int32_t               get_attn_rot_k_nrot() const;
    int32_t               get_attn_rot_v_nrot() const;
    const std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> & get_ctxs_bufs() const { return ctxs_bufs; }
    std::vector<ggml_context *> get_buffer_contexts() const;
    uint32_t              get_stream_for_seq(llama_seq_id seq_id) const;
    bool                  validate_seq_id(llama_seq_id seq_id) const;

    // Layer alias resolution
    uint32_t              get_owning_layer(uint32_t model_layer) const;

    // Hot cache capacity and capabilities
    uint32_t              get_hot_size() const;
    uint32_t              get_kv_hot_capacity() const override;
    bool                  can_use_legacy_attention() const override;
    bool                  is_xkv_bounded_hot() const override;
    std::shared_ptr<llama_xkv::xkv_hot_slot_pool> get_hot_slot_pool() const;

    // Apply a validated hot release plan after segment publish
    bool apply_hot_release_plan(
        const llama_xkv::xkv_hot_release_plan & plan,
        std::string * err = nullptr);

    // Atomic precommit for the seal path (replaces post-publish release).
    // Preflight: the plan triple is validated read-only against pool and
    // store (presence, generation, seal-lock). Commit: the pool triple is
    // validated + released inside one pool lock. After success, the store
    // publish of the same payloads is guaranteed no-fail with respect to the
    // hot rows, so no separate fallible post-publish pool step may run.
    // The runtime must call this BEFORE seal/publish while the rows are still
    // pool-bound; tensor data stays resident until reused, and maintain runs
    // with no interleaving reservation. SHADOW and unbounded caches never
    // release: they return true immediately without touching pool or store.
    bool precommit_hot_release(
        const llama_xkv::xkv_hot_release_plan & plan,
        std::string * err = nullptr);

    // Seal-gate variant of precommit_hot_release for invocation while the
    // store holds its own lock (or a reserved seal nonce): performs ZERO
    // store calls and only takes the pool lock, so it cannot deadlock on the
    // non-recursive store mutex. Lock order is store -> pool; no path in this
    // file ever takes the pool lock while holding the store lock and then
    // calls back into the store. The store must prevalidate under its own
    // lock, reserve the seal nonce (blocking concurrent mutation of these
    // payloads), invoke this, then relock/verify the nonce and commit
    // allocation-free. SHADOW/unbounded early-true as above.
    bool precommit_hot_release_nostore(
        const llama_xkv::xkv_hot_release_plan & plan,
        std::string * err = nullptr);

    // Seal-protocol split halves (see precommit docs). validate_hot_release is
    // the read-only pre-seal stage: triple shape, pool binding+generation,
    // store presence+generation, and unlocked pre-seal rows. Zero mutation;
    // never call it while holding the store lock (it locks internally).
    // commit_hot_release is the gate callback: SHADOW/unbounded early-true,
    // otherwise pool release only with zero store calls, safe under the
    // store lock. Invoke at most once per seal, only after a successful
    // validate with unbroken quiescence; gate-false/throw leaves every
    // subsystem unmutated (pool release is single-lock atomic).
    bool validate_hot_release(
        const llama_xkv::xkv_hot_release_plan & plan,
        std::string * err = nullptr);
    bool commit_hot_release(
        const llama_xkv::xkv_hot_release_plan & plan,
        std::string * err = nullptr);

    // XKV shared store accessor
    std::shared_ptr<llama_xkv::llama_xkv_cache_store> get_xkv_store() const;
    void init_xkv_store(const llama_cparams & cparams);
    const llama_kv_cells & get_cells(llama_seq_id seq_id) const;

    bool has_cell_ext() const;

    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;

    // XKV runtime coordinator (nullptr for OFF/MTP or rejected configs).
    // Owns safe-boundary sealing, admission planning, and snapshot joins.
    llama_xkv::llama_xkv_runtime * get_xkv_runtime();
    const llama_xkv::llama_xkv_runtime * get_xkv_runtime() const;

    // Persistent shared backend executor per device/buft (cached, delegating to other if view)
    std::shared_ptr<struct ggml_backend> get_xkv_executor(ggml_backend_buffer_type_t buft = nullptr) const;

    // RERoT write tags are sequence-scoped control state. apply_ubatch() copies
    // the active tag into each newly allocated physical cell. Ordinary
    // sequences have a cleared tag and therefore retain stock behavior.
    bool rerot_set_write_tag(llama_seq_id seq_id, const llama_kv_rerot_meta & tag) override;
    void rerot_clear_write_tag(llama_seq_id seq_id) override;

    bool rerot_can_publish_run(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        size_t * count) const override;

    bool rerot_can_reclassify_run(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        llama_rerot_visibility expected,
        llama_rerot_visibility replacement,
        uint64_t publish_epoch,
        size_t * count) const override;

    // Atomically publish every resident cell of a pending logical run. Returns
    // the number of cells transitioned; zero means no matching pending cells.
    size_t rerot_publish_run(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        uint64_t publish_epoch) override;

    size_t rerot_reclassify_run(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        llama_rerot_visibility expected,
        llama_rerot_visibility replacement,
        uint64_t publish_epoch) override;

    bool rerot_can_add_run_ref(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        llama_seq_id seq_id,
        size_t * count) const override;

    size_t rerot_add_run_ref(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        llama_seq_id seq_id) override;

    // Physical lookup of one logical run across all streams. Appends
    // (stream, cell) pairs in (stream, cell) order and returns the count.
    // KV-internal: the server scheduler stores run ids only and must never
    // cache physical indices across compaction, eviction, or restore.
    size_t rerot_find_run_cells(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        std::vector<std::pair<uint32_t, uint32_t>> * out) const;

    // Attention-only archive freeze (§7.2) for one execution sequence: every
    // PUBLIC_LIVE resident cell of episode_id referenced by exec_seq gains
    // archive_seq as keeper, then exec_seq is fully released. No K/V data
    // moves and visibility metadata is untouched. The returned kept count
    // may be zero (e.g. purely private history) while exec_seq is still
    // released. Returns 0 without touching anything when the ids are
    // invalid, cross different streams, or address a view cache.
    bool rerot_can_freeze_to_archive(
        uint64_t episode_id,
        llama_seq_id exec_seq,
        llama_seq_id archive_seq,
        size_t * count) const;

    size_t rerot_freeze_to_archive(
        uint64_t episode_id,
        llama_seq_id exec_seq,
        llama_seq_id archive_seq);

    // v1 persistence guard (§5.5/§25): true when saving seq_id (or the whole
    // cache for -1) would silently drop RERoT classification, in which case
    // state_write() refuses with an explicit error instead. Ordinary,
    // untagged state never triggers this.
    bool rerot_blocks_state_save(llama_seq_id seq_id) const;

    bool rerot_set_reader_view(llama_seq_id seq_id, const llama_rerot_reader_state & view) override;
    void rerot_clear_reader_view(llama_seq_id seq_id) override;

    // Build the indexed, query-grouped layout consumed by the backend-neutral
    // RERoT attention op. The returned key indices address the K/V view for the
    // current unified stream.
    llama_rerot_attn_layout rerot_build_attn_layout(
        const llama_ubatch & ubatch,
        uint32_t n_kv) const;

    bool rerot_batch_active(const llama_ubatch & ubatch) const;

    // Resolve a logical PAC-DFS view after any TriAttention eviction or cache
    // compaction. Resident cells are ordered by the logical run sequence and
    // densely repacked in virtual address space.
    rerot_resolved_view rerot_resolve_view(const llama_rerot_reader_view & view) const;

    // ========================================================================
    // Bounded staging capture API for committed seal ranges (§5.3)
    // ========================================================================
    bool can_capture_prerope_range(
        llama_seq_id seq_id,
        uint32_t cell_start,
        uint32_t cell_count,
        std::string * reason = nullptr) const;

    //
    // FlashPrefill legal-fragment planning (CacheFragments)
    //
    // Owner-derived short-lived layout: legal fragments, query phase groups
    // (same structure as the existing graph RoPE input; K is never re-phased)
    // and deduped exact rows. The old indexed rerot_build_attn_layout() above
    // is preserved untouched as the dense/oracle OFF path; building this
    // layout never forces old entries generation.
    //
    // role uses the frozen llama_flashprefill_role values; only PREFILL
    // (ordinary) and REROT_TEACHER_FORCED are ever eligible, everything else
    // yields an ineligible layout carrying its dense reason (never sparse on
    // doubt). block_k/BN, causal, and capacities come from params (caller
    // maps policy/wire config); SWA/alibi/2-D capability gates are evaluated
    // here from owner state. out is always assigned (cleared first).
    // Returns true iff the layout is eligible. Hard errors set error and
    // return false; policy ineligibility is not an error.
    bool flashprefill_build_layout(
        const llama_ubatch & ubatch,
        int32_t role,
        const llama_flashprefill_layout_params & params,
        llama_flashprefill_layout & out,
        std::string * error = nullptr) const;

    // Owner cache-side mutation epoch. Bumped on every metadata mutation
    // (append, seq_rm/cp/keep/add/div, reclaim, compact, shift update,
    // restore, publish/reclassify/ref/freeze, view/tag writes) once the flash
    // path is active; OFF performs no counter writes at all. Layouts stamp
    // this value plus the per-stream CellGeneration pairs; any mismatch means
    // stale (fail-closed).
    uint64_t flashprefill_epoch() const { return fp_epoch; }

    // Lazily opt into CellGeneration tracking on every stream. Called from the
    // flash path only (first build); OFF never calls it, so OFF keeps zero
    // generation overhead. Idempotent.
    void flashprefill_enable_tracking() const;

    bool flashprefill_tracking_enabled() const;

    // Per-stream CellGeneration (stamp, generation) pairs for freshness keys.
    // Never exact-+1 compares; saturated (MAX) either means always-invalid.
    void flashprefill_cell_stamps(std::vector<llama_flashprefill_cell_stamp> & out) const;

    uint32_t flashprefill_n_streams() const;

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

    // Gemma4 MTP: one-token slot_info pointing at the last populated cell for seq_id (read-only graphs).
    slot_info mtp_slot_info(llama_seq_id seq_id) const;

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

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
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
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);
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

    // FlashPrefill planning cache (mutable so const graph KV inputs can build;
    // same discipline as rerot_layout above). Keyed by owned ubatch + role +
    // capacities; validated against owner stamps on every use.
    mutable bool fp_key_valid = false;
    mutable llama_flashprefill_build_key fp_key;
    mutable llama_flashprefill_layout fp_layout;
    mutable uint64_t fp_cells_epoch = 0;
    mutable std::vector<llama_flashprefill_cell_stamp> fp_cell_stamps;
};
