#include "llama-memory-recurrent.h"

#include "ggml-backend.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-batch.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>

//
// llama_memory_recurrent
//

llama_memory_recurrent::llama_memory_recurrent(
        const llama_model & model,
                ggml_type   type_r,
                ggml_type   type_s,
                     bool   offload,
                 uint32_t   mem_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                 uint32_t   n_brain_max,
                 uint32_t   n_hand_max,
    const layer_filter_cb & filter) : hparams(model.hparams), n_seq_max(n_seq_max) {
    const int32_t n_layer = hparams.n_layer();

    head = 0;
    size = mem_size;
    used = 0;

    this->n_rs_seq = n_rs_seq;

    const bool grouped = n_brain_max > 0 || n_hand_max > 0;
    if (grouped) {
        if (n_brain_max == 0 || n_hand_max == 0 || mem_size != n_hand_max) {
            throw std::invalid_argument("invalid grouped recurrent brain/hand capacity");
        }
        n_brain_rows = n_brain_max;
        n_hand_rows = n_hand_max;
        brain_capacity = n_brain_max;
        hand_capacity = n_hand_max;
    }

    // RERoT logical seq-id capacity (§6.5): the parked/recursive-fork lineage
    // may reference any id in [0, LLAMA_MAX_SEQ), while the physical state
    // tensors stay at mem_size cells. Only these logical maps grow; no tensor
    // is widened. Ordinary (RERoT OFF) ids are unaffected.
    rs_idx.assign(LLAMA_MAX_SEQ, 0);

    cells.clear();
    cells.resize(mem_size);
    tails.assign(LLAMA_MAX_SEQ, -1);
    seq_brain.assign(LLAMA_MAX_SEQ, -1);
    seq_episode.assign(LLAMA_MAX_SEQ, 0);
    seq_node.assign(LLAMA_MAX_SEQ, LLAMA_REROT_NODE_INVALID);
    seq_public_write.assign(LLAMA_MAX_SEQ, 0);
    brain_episode.assign(n_brain_rows, 0);

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                // r, s, d per layer, plus the separate PLE conv row where the model has one
                /*.mem_size   =*/ size_t((hparams.ple_conv_state() > 0 ? 4u : 3u)*n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    r_l.resize(n_layer);
    s_l.resize(n_layer);
    p_l.resize(n_layer);
    d_l.resize(n_layer);
    s_shared_l.assign(n_layer, 0);

    uint32_t recurrent_ordinal = 0;
    for (int i = 0; i < n_layer; i++) {
        if (filter && !filter(i)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: skipped\n", __func__, i);
            continue;
        }

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(i);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s, layer %3d: dev = %s\n", __func__, i, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for rs cache");
        }

        const bool shared_s = grouped && recurrent_ordinal >= 3;
        const uint32_t r_rows = mem_size * (1 + n_rs_seq);
        // Shared layers keep current public/private brain rows. Rollback only
        // applies to public generation, so its additional planes hold B rows,
        // not both public and planner brains.
        const uint32_t s_rows = shared_s
            ? 2 * n_brain_rows + n_brain_rows * n_rs_seq
            : mem_size * (1 + n_rs_seq);
        ggml_tensor * r = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), r_rows);
        ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), s_rows);
        ggml_tensor * d = shared_s
            ? ggml_new_tensor_2d(
                // This is persistent recurrent state, not a lossy KV cache.
                // F16 overlays round every child transition even when DDVR
                // inputs are identical to native; one-step output tests miss
                // the loss because output precedes this store.
                ctx, GGML_TYPE_F32, hparams.n_embd_s(),
                mem_size * (1 + n_rs_seq))
            : nullptr;
        ggml_format_name(r, "cache_r_l%d", i);
        ggml_format_name(s, "cache_s_l%d", i);
        if (d) {
            ggml_format_name(d, "cache_d_l%d", i);
        }
        r_l[i] = r;
        s_l[i] = s;
        // the PLE history needs its own row
        if (hparams.ple_conv_state() > 0 && hparams.is_ple(i)) {
            ggml_tensor * p = ggml_new_tensor_2d(ctx, type_r, hparams.ple_conv_state(), r_rows);
            ggml_format_name(p, "cache_ple_r_l%d", i);
            p_l[i] = p;
        }

        d_l[i] = d;
        s_shared_l[i] = shared_s;
        ++recurrent_ordinal;
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for rs cache");
        }
        ggml_backend_buffer_clear(buf, 0);
        LLAMA_LOG_INFO("%s: %10s RS buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    {
        const size_t memory_size_r = size_r_bytes();
        const size_t memory_size_s = size_s_bytes();
        const size_t memory_size_d = size_d_bytes();
        const size_t memory_size_p = size_p_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u hand rows, %3d layers, %2u seqs %2u rs_seq, %u brain rows), R (%s): %7.2f MiB, S (%s): %7.2f MiB, hand state (f32): %7.2f MiB, P (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_r + memory_size_s + memory_size_d + memory_size_p) / (1024.0f * 1024.0f), mem_size, n_layer, n_seq_max, n_rs_seq, n_brain_rows,
                ggml_type_name(type_r), (float)memory_size_r / (1024.0f * 1024.0f),
                ggml_type_name(type_s), (float)memory_size_s / (1024.0f * 1024.0f),
                (float)memory_size_d / (1024.0f * 1024.0f),
                ggml_type_name(type_r), (float)memory_size_p / (1024.0f * 1024.0f));
    }
}

void llama_memory_recurrent::clear(bool data) {
    for (int32_t i = 0; i < (int32_t) size; ++i) {
        cells[i].pos = -1;
        cells[i].seq_id.clear();
        cells[i].src = -1;
    }
    std::fill(tails.begin(), tails.end(), -1);
    std::fill(seq_brain.begin(), seq_brain.end(), -1);
    std::fill(seq_episode.begin(), seq_episode.end(), 0);
    std::fill(seq_node.begin(), seq_node.end(), LLAMA_REROT_NODE_INVALID);
    std::fill(seq_public_write.begin(), seq_public_write.end(), 0);
    episode_brain.clear();
    std::fill(brain_episode.begin(), brain_episode.end(), 0);

    head = 0;
    used = 0;

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    } else {
        // Clear brain rows and hand rows deterministically even when full buffer clear is false
        for (uint32_t b = 0; b < n_brain_rows; ++b) {
            clear_brain_row((int32_t) b);
        }
        for (uint32_t h = 0; h < size; ++h) {
            clear_hand_row((int32_t) h);
        }
    }

    std::fill(rs_idx.begin(), rs_idx.end(), 0);
}

bool llama_memory_recurrent::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    uint32_t new_head = size;

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    const bool rm_all = p0 == 0 && p1 == std::numeric_limits<llama_pos>::max();
    if (rm_all) {
        if (seq_id >= 0) {
            set_rs_idx(seq_id, 0);
            const int32_t b_row = brain_row_for_seq(seq_id);
            if (b_row >= 0 && (size_t) seq_id < seq_episode.size() && seq_episode[(size_t) seq_id] == 0) {
                clear_brain_row(b_row);
            }
            if ((size_t) seq_id < seq_brain.size()) {
                seq_brain[(size_t) seq_id] = -1;
                seq_episode[(size_t) seq_id] = 0;
                seq_node[(size_t) seq_id] = LLAMA_REROT_NODE_INVALID;
                seq_public_write[(size_t) seq_id] = 0;
            }
        } else {
            std::fill(rs_idx.begin(), rs_idx.end(), 0);
            std::fill(tails.begin(), tails.end(), -1);
            std::fill(seq_brain.begin(), seq_brain.end(), -1);
            std::fill(seq_episode.begin(), seq_episode.end(), 0);
            std::fill(seq_node.begin(), seq_node.end(), LLAMA_REROT_NODE_INVALID);
            std::fill(seq_public_write.begin(), seq_public_write.end(), 0);
            for (uint32_t b = 0; b < n_brain_rows; ++b) {
                clear_brain_row((int32_t) b);
            }
            for (uint32_t h = 0; h < size; ++h) {
                clear_hand_row((int32_t) h);
            }
        }
    }

    // models like Mamba or RWKV can't have a state partially erased at the end
    // of the sequence because their state isn't preserved for previous tokens
    // NOTE: gated on the logical capacity (LLAMA_MAX_SEQ), not the configured
    // n_seq_max, so parked RERoT lineages remain addressable. Physical cells
    // are still bounded by `size`.
    if (seq_id >= (int64_t) LLAMA_MAX_SEQ) {
        return false;
    }
    if (0 <= seq_id) {
        int32_t & tail_id = tails[(size_t) seq_id];
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];

            // partial rollback via per-token snapshot index (bounded by n_rs_seq)
            if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
                const llama_pos rollback = cell.pos - (p0 - 1);
                if (rollback >= 1 && rollback <= (llama_pos) n_rs_seq) {
                    set_rs_idx(seq_id, (uint32_t) rollback);
                    cell.pos = p0 - 1;
                    return true;
                }
                return false;
            }
            // invalidate tails which will be cleared
            if (p0 <= cell.pos && cell.pos < p1) {
                tail_id = -1;
            }
        }
    } else {
        // seq_id is negative, then the range should include everything or nothing
        if (p0 != p1 && (p0 != 0 || p1 != std::numeric_limits<llama_pos>::max())) {
            //printf("[DEBUG] inside `llama_memory_recurrent::seq_rm`: `seq_id` is negative, so returning false\n");
            return false;
        }
    }

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].pos >= p0 && cells[i].pos < p1) {
            if (seq_id < 0) {
                cells[i].seq_id.clear();
            } else if (cells[i].has_seq_id(seq_id)) {
                cells[i].seq_id.erase(seq_id);
            } else {
                continue;
            }
            if (cells[i].is_empty()) {
                // keep count of the number of used cells
                if (cells[i].pos >= 0) {
                    used--;
                }
                cells[i].pos = -1;
                cells[i].src = -1;
                clear_hand_row((int32_t) i);
                if (new_head == size) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }

    return true;
}

void llama_memory_recurrent::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (seq_id_src == seq_id_dst) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // RERoT parking: the child only joins the parent tail cell's ref set, so
    // many parked children share one tail with no tensor copy. The first real
    // write after admission allocates an exclusive cell in find_slot() (COW).
    // Releasing an exec id later drops just its own ref, leaving parked
    // siblings (and recursive-fork grandchildren) undisturbed.
    if ((uint32_t) seq_id_dst < LLAMA_MAX_SEQ && (uint32_t) seq_id_src < LLAMA_MAX_SEQ) {
        seq_brain[(size_t) seq_id_dst] = brain_row_for_seq(seq_id_src);
        seq_episode[(size_t) seq_id_dst] = seq_episode[(size_t) seq_id_src];
        seq_node[(size_t) seq_id_dst] = seq_node[(size_t) seq_id_src];
        seq_public_write[(size_t) seq_id_dst] = seq_public_write[(size_t) seq_id_src];

        int32_t & tail_src = tails[(size_t) seq_id_src];
        int32_t & tail_dst = tails[(size_t) seq_id_dst];
        if (tail_dst >= 0) {
            auto & cell_dst = cells[(size_t) tail_dst];

            cell_dst.seq_id.erase(seq_id_dst);
            tail_dst = -1;
            if (cell_dst.seq_id.empty()) {
                cell_dst.pos = -1;
                cell_dst.src = -1;
                used -= 1;
            }
        }
        if (tail_src >= 0) {
            auto & cell_src = cells[(size_t) tail_src];

            cell_src.seq_id.insert(seq_id_dst);
            tail_dst = tail_src;
        }
    }
}

bool llama_memory_recurrent::seq_rm_attention(
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1) {
    // recurrent-only memory holds no attention state: the attention-side op is
    // vacuously successful so uniform two-sided release paths keep working.
    GGML_UNUSED(seq_id);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    return true;
}

void llama_memory_recurrent::seq_cp_attention(
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos p0,
        llama_pos p1) {
    GGML_UNUSED(seq_id_src);
    GGML_UNUSED(seq_id_dst);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
}

bool llama_memory_recurrent::seq_rm_recurrent(
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1) {
    return seq_rm(seq_id, p0, p1);
}

void llama_memory_recurrent::seq_cp_recurrent(
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos p0,
        llama_pos p1) {
    seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_recurrent::seq_keep(llama_seq_id seq_id) {
    uint32_t new_head = size;

    for (size_t i = 0; i < tails.size(); ++i) {
        if ((llama_seq_id) i != seq_id) {
            tails[i] = -1;
        }
    }

    for (uint32_t i = 0; i < size; ++i) {
        if (!cells[i].has_seq_id(seq_id)) {
            if (cells[i].pos >= 0) {
                used--;
            }

            cells[i].pos = -1;
            cells[i].src = -1;
            cells[i].seq_id.clear();
            clear_hand_row((int32_t) i);

            if (new_head == size) {
                new_head = i;
            }
        } else {
            cells[i].seq_id.clear();
            cells[i].seq_id.insert(seq_id);
            if (seq_id >= 0 && (size_t) seq_id < tails.size()) {
                tails[(size_t) seq_id] = (int32_t) i;
            }
        }
    }

    if (new_head != size && new_head < head) {
        head = new_head;
    }
}

void llama_memory_recurrent::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (shift == 0) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the
    if (p0 == p1) {
        return;
    }

    // for Mamba-like or RWKV models, only the pos needs to be shifted
    if (0 <= seq_id && seq_id < (int64_t) LLAMA_MAX_SEQ) {
        const int32_t tail_id = tails[(size_t) seq_id];
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos += shift;
            }
        }
    }
}

void llama_memory_recurrent::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    // for Mamba-like or RWKV models, only the pos needs to be changed
    if (0 <= seq_id && seq_id < (int64_t) LLAMA_MAX_SEQ) {
        const int32_t tail_id = tails[(size_t) seq_id];
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos /= d;
            }
        }
    }
}

llama_pos llama_memory_recurrent::seq_pos_min(llama_seq_id seq_id) const {
    llama_pos result = std::numeric_limits<llama_pos>::max();

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::min(result, cells[i].pos);
        }
    }

    if (result == std::numeric_limits<llama_pos>::max()) {
        result = -1;
    }

    return result;
}

llama_pos llama_memory_recurrent::seq_pos_max(llama_seq_id seq_id) const {
    llama_pos result = -1;

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::max(result, cells[i].pos);
        }
    }

    return result;
}

void llama_memory_recurrent::set_rs_idx(llama_seq_id seq_id, uint32_t idx) {
    if (seq_id < 0 || (size_t) seq_id >= rs_idx.size()) {
        return;
    }
    rs_idx[seq_id] = (idx > n_rs_seq) ? n_rs_seq : idx;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_recurrent::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [_, buf] : ctxs_bufs) {
        ret[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
    }
    return ret;
}

llama_memory_context_ptr llama_memory_recurrent::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // TODO: non-sequential equal split can be done if using unified KV cache
                //       for simplicity, we always use sequential equal split for now
                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                ubatch = balloc.split_equal(n_ubatch, true, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        if (!prepare(ubatches)) {
            break;
        }

        return std::make_unique<llama_memory_recurrent_context>(this, std::move(ubatches));
    } while (false);

    return std::make_unique<llama_memory_recurrent_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_recurrent::init_full() {
    return std::make_unique<llama_memory_recurrent_context>(this);
}

llama_memory_context_ptr llama_memory_recurrent::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);

    return std::make_unique<llama_memory_recurrent_context>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

bool llama_memory_recurrent::prepare(const std::vector<llama_ubatch> & ubatches) {
    // simply remember the full state because it is very small for this type of cache
    // TODO: optimize
    auto org_cells = cells;
    auto org_tails = tails;
    auto org_used = used;
    auto org_head = head;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        if (!find_slot(ubatch)) {
            success = false;
            break;
        }
    }

    // restore the original state
    cells = std::move(org_cells);
    tails = std::move(org_tails);
    used = org_used;
    head = org_head;

    return success;
}

bool llama_memory_recurrent::find_slot(const llama_ubatch & ubatch) {
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t hand_begin   = 0;
    const uint32_t hand_end     = size;
    const uint32_t hand_size    = size;

    // Persistent brain rows live in separate S tensors. The ordinary cell
    // allocator owns only hand rows.
    if (head >= hand_end || head > used + 2*n_seqs) {
        head = 0;
    }

    // For recurrent state architectures (like Mamba or RWKV),
    // each cache cell can store the state for a whole sequence.
    // A slot should be always be contiguous.

    // can only process batches with an equal number of new tokens in each sequence
    GGML_ASSERT(ubatch.equal_seqs());

    int32_t min = hand_end - 1;
    int32_t max = hand_begin;

    // everything should fit if all seq_ids are smaller than the max
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens; // first token of sequence set s
        const uint32_t n_seq_id = ubatch.n_seq_id[i];

        for (uint32_t j = 0; j < n_seq_id; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];

            if (seq_id < 0 || (uint32_t) seq_id >= LLAMA_MAX_SEQ) {
                LLAMA_LOG_ERROR("%s: seq_id=%d >= LLAMA_MAX_SEQ=%u\n", __func__, seq_id, (unsigned) LLAMA_MAX_SEQ);
                return false;
            }
            if (j > 0) {
                int32_t & seq_tail = tails[(size_t) seq_id];
                if (seq_tail >= 0) {
                    auto & cell = cells[(size_t) seq_tail];
                    cell.seq_id.erase(seq_id);
                    seq_tail = -1;
                    if (cell.seq_id.empty()) {
                        cell.pos = -1;
                        cell.src = -1;
                        used -= 1;
                    }
                }
            }
        }
    }

#ifndef NDEBUG
    {
        std::vector<int32_t> tails_verif;
        tails_verif.assign(tails.size(), -1);
        for (uint32_t i = 0; i < size; ++i) {
            auto & cell = cells[i];
            for (llama_seq_id seq_id : cell.seq_id) {
                if (tails_verif[seq_id] != -1) {
                    LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tails_verif[seq_id]);
                }
                tails_verif[seq_id] = i;
            }
        }
        for (uint32_t i = 0; i < tails.size(); ++i) {
            if (tails_verif[i] != tails[i]) {
                LLAMA_LOG_ERROR("%s: wrong tail for seq_id %d, (%d instead of %d)\n", __func__, i, tails[i], tails_verif[i]);
            }
        }
    }
#endif

    uint32_t n_needed = 0;
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        const int32_t tail = tails[(size_t) seq_id];
        if (tail < 0 || cells[(size_t) tail].seq_id.size() != 1) {
            ++n_needed;
        }
    }
    if (n_needed > hand_size - used) {
        return false;
    }

    // find next empty hand cell
    uint32_t next_empty_cell = head;

    for (uint32_t i = 0; i < hand_size; ++i) {
        if (next_empty_cell >= hand_end) { next_empty_cell = hand_begin; }
        auto & cell = cells[next_empty_cell];
        if (cell.is_empty()) { break; }
        next_empty_cell += 1;
    }

    // find usable cell range
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        int32_t & seq_tail = tails[(size_t) seq_id];
        bool has_cell = false;
        if (seq_tail >= 0) {
            auto & cell = cells[(size_t) seq_tail];
            GGML_ASSERT(cell.has_seq_id(seq_id));
            if (cell.seq_id.size() == 1) { has_cell = true; }
        }
        if (!has_cell) {
            auto & empty_cell = cells[next_empty_cell];
            GGML_ASSERT(empty_cell.is_empty());
            if (seq_tail >= 0) {
                auto & orig_cell = cells[(size_t) seq_tail];
                empty_cell.pos = orig_cell.pos;
                empty_cell.src = orig_cell.src;
                orig_cell.seq_id.erase(seq_id);
                empty_cell.seq_id.insert(seq_id);
                GGML_ASSERT(!orig_cell.is_empty());
            }
            seq_tail = next_empty_cell;
            if (s + 1 < n_seqs) {
                for (uint32_t j = 0; j < hand_size; ++j) {
                    next_empty_cell += 1;
                    if (next_empty_cell >= hand_end) { next_empty_cell = hand_begin; }
                    auto & cell = cells[next_empty_cell];
                    if (cell.is_empty()) { break; }
                }
            }
        }
        if (min > seq_tail) { min = seq_tail; }
        if (max < seq_tail) { max = seq_tail; }
    }

    // gather and re-order
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const int32_t dst_id = s + min;
        const int32_t src_id = tails[(size_t) ubatch.seq_id[i][0]];
        if (dst_id != src_id) {
            auto & dst_cell = cells[dst_id];
            auto & src_cell = cells[src_id];

            std::swap(dst_cell.pos, src_cell.pos);
            std::swap(dst_cell.src, src_cell.src);
            std::swap(dst_cell.seq_id, src_cell.seq_id);

            // swap logical-to-physical mappings
            for (size_t j = 0; j < tails.size(); ++j) {
                int32_t & tail = tails[j];
                if (tail == src_id) {
                    tail = dst_id;
                } else if (tail == dst_id) {
                    tail = src_id;
                }
            }
        }
    }

    // update the pos of the used seqs
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const llama_pos last_pos = ubatch.pos[i + n_seq_tokens - 1];
        const int32_t cell_id = s + min;
        auto & cell = cells[cell_id];

        if (cell.pos >= 0 && last_pos != cell.pos + (llama_pos) n_seq_tokens) {
            // What should happen when the pos backtracks or skips a value?
            // Clearing the state mid-batch would require special-casing which isn't done.
            LLAMA_LOG_WARN("%s: non-consecutive token position %d after %d for sequence %d with %u new tokens\n",
                __func__, last_pos, cell.pos, ubatch.seq_id[i][0], n_seq_tokens);
        }
        cell.pos = last_pos;
        cell.seq_id.clear();
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];
            cell.seq_id.insert(seq_id);
            tails[(size_t) seq_id] = cell_id;
        }
    }

    // Find first cell without src refs, to use as the zero-ed state
    {
        // TODO: bake-in src refcounts in the cell metadata
        std::vector<int32_t> refcounts(size, 0);
        for (size_t i = 0; i < size; ++i) {
            const int32_t src = cells[i].src;
            if (src >= 0) {
                refcounts[src] += 1;
            }
        }

        rs_z = -1;
        for (int i = min; i <= max; ++i) {
            if (refcounts[i] == 0) {
                rs_z = i;
                break;
            }
        }

        for (int i = min; i <= max; ++i) {
            if (cells[i].src < 0) {
                GGML_ASSERT(rs_z >= 0);
                cells[i].src0 = rs_z;
            } else {
                // Stage the source ids for all used cells to allow correct seq_* behavior
                // and still make these values available when setting the inputs
                cells[i].src0 = cells[i].src;
            }
            cells[i].src = i; // avoid moving or clearing twice
        }
    }

    // allow getting the range of used cells, from head to head + n
    head = min;
    n    = max - min + 1;
    used = std::count_if(cells.begin(), cells.end(),
        [](const mem_cell & cell){ return !cell.is_empty(); });

    // sanity check
    return n >= n_seqs;
}

bool llama_memory_recurrent::get_can_shift() const {
    return true;
}

uint32_t llama_memory_recurrent::get_recurrent_capacity() const {
    return size;
}

uint32_t llama_memory_recurrent::get_recurrent_used() const {
    return used;
}

void llama_memory_recurrent::set_grouped_layout(uint32_t n_brains, uint32_t n_hands) {
    GGML_ASSERT(n_brains > 0);
    GGML_ASSERT(n_hands > 0);
    GGML_ASSERT(n_brains == n_brain_rows);
    GGML_ASSERT(n_hands == n_hand_rows);
    GGML_ASSERT(n_hands == size);
    GGML_ASSERT(used == 0);
}

bool llama_memory_recurrent::is_s_shared(int32_t il) const {
    return il >= 0 && (size_t) il < s_shared_l.size() && s_shared_l[(size_t) il] != 0;
}

uint32_t llama_memory_recurrent::get_brain_capacity() const {
    return is_grouped_layout() ? brain_capacity : get_recurrent_capacity();
}

uint32_t llama_memory_recurrent::get_hand_capacity() const {
    return is_grouped_layout() ? hand_capacity : get_recurrent_capacity();
}

uint32_t llama_memory_recurrent::get_brain_used() const {
    if (!is_grouped_layout()) {
        return get_recurrent_used();
    }
    return (uint32_t) std::count_if(
        brain_episode.begin(), brain_episode.end(),
        [](uint64_t episode_id) { return episode_id != 0; });
}

uint32_t llama_memory_recurrent::get_hand_used() const {
    return get_recurrent_used();
}

int32_t llama_memory_recurrent::brain_row_for_seq(llama_seq_id seq_id) const {
    if (!is_grouped_layout() || seq_id < 0 || (size_t) seq_id >= seq_brain.size()) {
        return -1;
    }
    if (seq_brain[(size_t) seq_id] >= 0) {
        return seq_brain[(size_t) seq_id];
    }

    // Before an episode is armed, root request sequences occupy the first B
    // server slots. Their stable slot id is therefore their provisional brain
    // row during prompt prefill.
    return seq_id < (llama_seq_id) n_brain_rows ? seq_id : -1;
}

int32_t llama_memory_recurrent::brain_read_row_for_seq(llama_seq_id seq_id) const {
    int32_t row = brain_row_for_seq(seq_id);
    if (row < 0) {
        return -1;
    }

    const bool private_planner =
        seq_episode[(size_t) seq_id] != 0 &&
        seq_public_write[(size_t) seq_id] == 0 &&
        seq_node[(size_t) seq_id] == 0;
    if (private_planner) {
        row += (int32_t) n_brain_rows;
    }

    const uint32_t snapshot =
        seq_id >= 0 && (size_t) seq_id < rs_idx.size()
            ? rs_idx[(size_t) seq_id]
            : 0;
    // A default child never committed the shared-brain snapshot slots. Its
    // rollback selector belongs to its private hand, not to the root's (often
    // unwritten) historical brain slots. Pair the saved hand with the same
    // unchanged public brain used when it was produced.
    if (snapshot == 0 || private_planner || uses_native_child_state(seq_id)) {
        return row;
    }
    return (int32_t) (
        2 * n_brain_rows +
        (snapshot - 1) * n_brain_rows +
        (uint32_t) row);
}

int32_t llama_memory_recurrent::acquire_brain_row(
        uint64_t episode_id,
        llama_seq_id seq_id) {
    if (!is_grouped_layout() || episode_id == 0 ||
        seq_id < 0 || (size_t) seq_id >= seq_brain.size()) {
        return -1;
    }

    const auto found = episode_brain.find(episode_id);
    if (found != episode_brain.end()) {
        seq_brain[(size_t) seq_id] = found->second;
        seq_episode[(size_t) seq_id] = episode_id;
        return found->second;
    }

    int32_t row = brain_row_for_seq(seq_id);
    if (row < 0 || brain_episode[(size_t) row] != 0) {
        const auto free = std::find(brain_episode.begin(), brain_episode.end(), 0);
        if (free == brain_episode.end()) {
            return -1;
        }
        row = (int32_t) std::distance(brain_episode.begin(), free);
    }

    brain_episode[(size_t) row] = episode_id;
    episode_brain.emplace(episode_id, row);
    seq_brain[(size_t) seq_id] = row;
    seq_episode[(size_t) seq_id] = episode_id;
    return row;
}

void llama_memory_recurrent::clear_brain_row(int32_t brain_row) {
    if (brain_row < 0 || (uint32_t) brain_row >= n_brain_rows) {
        return;
    }

    for (size_t il = 0; il < s_l.size(); ++il) {
        ggml_tensor * s = s_l[il];
        if (!s || !is_s_shared((int32_t) il)) {
            continue;
        }
        const size_t row_size = ggml_row_size(s->type, s->ne[0]);
        std::vector<uint8_t> zero(row_size, 0);
        ggml_backend_tensor_set(
            s, zero.data(), (uint32_t) brain_row * row_size, row_size);
        ggml_backend_tensor_set(
            s, zero.data(),
            (n_brain_rows + (uint32_t) brain_row) * row_size,
            row_size);
        for (uint32_t snapshot = 1; snapshot <= n_rs_seq; ++snapshot) {
            const size_t row =
                2 * (size_t) n_brain_rows +
                (size_t) (snapshot - 1) * n_brain_rows +
                (uint32_t) brain_row;
            ggml_backend_tensor_set(
                s, zero.data(), row * row_size, row_size);
        }
    }
    if (backend_sched) {
        ggml_backend_sched_synchronize(backend_sched);
    }
}

void llama_memory_recurrent::clear_hand_row(int32_t hand_row) {
    if (hand_row < 0 || (uint32_t) hand_row >= size) {
        return;
    }

    for (size_t il = 0; il < r_l.size(); ++il) {
        ggml_tensor * r = r_l[il];
        if (r) {
            const size_t row_size = ggml_row_size(r->type, r->ne[0]);
            std::vector<uint8_t> zero(row_size, 0);
            for (uint32_t snapshot = 0; snapshot <= n_rs_seq; ++snapshot) {
                const size_t row = (size_t) snapshot * size + (uint32_t) hand_row;
                ggml_backend_tensor_set(r, zero.data(), row * row_size, row_size);
            }
        }
    }

    for (size_t il = 0; il < s_l.size(); ++il) {
        if (is_s_shared((int32_t) il)) {
            ggml_tensor * d = d_l[il];
            if (d) {
                const size_t row_size = ggml_row_size(d->type, d->ne[0]);
                std::vector<uint8_t> zero(row_size, 0);
                for (uint32_t snapshot = 0; snapshot <= n_rs_seq; ++snapshot) {
                    const size_t row = (size_t) snapshot * size + (uint32_t) hand_row;
                    ggml_backend_tensor_set(d, zero.data(), row * row_size, row_size);
                }
            }
        } else {
            ggml_tensor * s = s_l[il];
            if (s) {
                const size_t row_size = ggml_row_size(s->type, s->ne[0]);
                std::vector<uint8_t> zero(row_size, 0);
                for (uint32_t snapshot = 0; snapshot <= n_rs_seq; ++snapshot) {
                    const size_t row = (size_t) snapshot * size + (uint32_t) hand_row;
                    ggml_backend_tensor_set(s, zero.data(), row * row_size, row_size);
                }
            }
        }
    }
    if (backend_sched) {
        ggml_backend_sched_synchronize(backend_sched);
    }
}

bool llama_memory_recurrent::uses_native_child_state(llama_seq_id seq_id) const {
    if (!is_grouped_layout() || seq_id < 0 || (size_t) seq_id >= seq_node.size() ||
        seq_episode[(size_t) seq_id] == 0 || seq_node[(size_t) seq_id] == 0 ||
        seq_node[(size_t) seq_id] == LLAMA_REROT_NODE_INVALID) {
        return false;
    }
    const char * mode = std::getenv("LLAMA_REROT_RBB_ABLATION");
    return !mode || (std::strcmp(mode, "shared-rbb") != 0 && std::strcmp(mode, "raw-redundant") != 0);
}

bool llama_memory_recurrent::rerot_set_write_tag(
        llama_seq_id seq_id,
        const llama_kv_rerot_meta & tag) {
    if (!is_grouped_layout()) {
        return true;
    }
    if (seq_id < 0 || (size_t) seq_id >= seq_node.size()) {
        return false;
    }
    const bool same_root = seq_episode[(size_t) seq_id] == tag.episode_id &&
        seq_node[(size_t) seq_id] == 0 && tag.node_id == 0;
    const bool was_private = seq_public_write[(size_t) seq_id] == 0;
    const bool now_private = tag.visibility != llama_rerot_visibility::public_live;
    const int32_t row = acquire_brain_row(tag.episode_id, seq_id);
    if (row < 0) {
        return false;
    }
    if (same_root && was_private != now_private && tails[(size_t) seq_id] >= 0) {
        // Changing B is a coordinate change, NOT a recurrent transition:
        // B_old + H_old = B_new + H_new. capture_hand_seed expresses the
        // effective local state relative to the current PUBLIC brain already.
        auto seed = capture_hand_seed(0, seq_id);
        if (!seed) {
            return false;
        }
        if (now_private) {
            for (size_t il = 0; il < s_l.size(); ++il) {
                if (!is_s_shared((int32_t) il) || !s_l[il]) {
                    continue;
                }
                const size_t n = (size_t) s_l[il]->ne[0];
                std::vector<float> public_brain(n), private_brain(n);
                std::vector<float> hand(n);
                ggml_backend_tensor_get(s_l[il], public_brain.data(), (size_t) row * n * sizeof(float), n * sizeof(float));
                ggml_backend_tensor_get(s_l[il], private_brain.data(),
                    ((size_t) n_brain_rows + row) * n * sizeof(float), n * sizeof(float));
                std::memcpy(hand.data(), seed->state_bytes[il].data(), n * sizeof(float));
                for (size_t i = 0; i < n; ++i) {
                    hand[i] = hand[i] + public_brain[i] - private_brain[i];
                }
                std::memcpy(seed->state_bytes[il].data(), hand.data(), n * sizeof(float));
            }
        }
        if (!apply_hand_seed(seq_id, seed)) {
            return false;
        }
    }
    seq_node[(size_t) seq_id] = tag.node_id;
    seq_public_write[(size_t) seq_id] =
        tag.visibility == llama_rerot_visibility::public_live ? 1 : 0;
    return true;
}

void llama_memory_recurrent::rerot_clear_write_tag(llama_seq_id seq_id) {
    if (seq_id < 0 || (size_t) seq_id >= seq_public_write.size()) {
        return;
    }
    seq_public_write[(size_t) seq_id] = 0;
}

void llama_memory_recurrent::rerot_release_episode(uint64_t episode_id) {
    const auto found = episode_brain.find(episode_id);
    if (found == episode_brain.end()) {
        return;
    }

    const int32_t brain_row = found->second;
    for (size_t i = 0; i < seq_episode.size(); ++i) {
        if (seq_episode[i] == episode_id) {
            seq_episode[i] = 0;
            seq_node[i] = LLAMA_REROT_NODE_INVALID;
            seq_brain[i] = -1;
            seq_public_write[i] = 0;

            if (i < tails.size()) {
                const int32_t tail_id = tails[i];
                if (tail_id >= 0 && (size_t) tail_id < cells.size()) {
                    auto & cell = cells[(size_t) tail_id];
                    cell.seq_id.erase((llama_seq_id) i);
                    if (cell.is_empty()) {
                        if (cell.pos >= 0) {
                            used--;
                        }
                        cell.pos = -1;
                        cell.src = -1;
                    }
                    clear_hand_row(tail_id);
                }
                tails[i] = -1;
            }
            if (i < rs_idx.size()) {
                rs_idx[i] = 0;
            }
        }
    }
    episode_brain.erase(found);
    brain_episode[(size_t) brain_row] = 0;
    clear_brain_row(brain_row);
}

std::shared_ptr<llama_memory_recurrent::hand_seed> llama_memory_recurrent::capture_hand_seed(
        uint64_t fork_id,
        llama_seq_id source_seq) {
    if (source_seq < 0 || (size_t) source_seq >= tails.size() || tails[(size_t) source_seq] < 0) {
        return nullptr;
    }

    const int32_t tail = tails[(size_t) source_seq];
    const int32_t row = cells[(size_t) tail].src >= 0 ? cells[(size_t) tail].src : tail;
    const uint32_t snapshot = rs_idx[(size_t) source_seq];
    const size_t local_row = (size_t) snapshot * size + (uint32_t) row;
    auto seed = std::make_shared<hand_seed>();
    seed->fork_id = fork_id;
    seed->source_hand_row = row;
    seed->source_pos = cells[(size_t) tail].pos;
    seed->conv_tail_bytes.resize(r_l.size());
    seed->state_bytes.resize(s_l.size());

    auto read_row = [&](ggml_tensor * tensor, std::vector<uint8_t> & bytes) {
        const size_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);
        bytes.resize(row_size);
        ggml_backend_t backend = backend_sched
            ? ggml_backend_sched_get_tensor_backend(backend_sched, tensor)
            : nullptr;
        if (backend) {
            ggml_backend_tensor_get_async(
                backend,
                tensor,
                bytes.data(),
                local_row * row_size,
                row_size);
        } else {
            ggml_backend_tensor_get(
                tensor,
                bytes.data(),
                local_row * row_size,
                row_size);
        }
    };

    for (size_t il = 0; il < r_l.size(); ++il) {
        if (r_l[il]) {
            read_row(r_l[il], seed->conv_tail_bytes[il]);
        }
    }
    const int32_t brain_row = brain_row_for_seq(source_seq);
    const bool private_planner =
        is_grouped_layout() &&
        brain_row >= 0 &&
        seq_episode[(size_t) source_seq] != 0 &&
        seq_node[(size_t) source_seq] == 0 &&
        seq_public_write[(size_t) source_seq] == 0;

    for (size_t il = 0; il < s_l.size(); ++il) {
        if (!s_l[il]) {
            continue;
        }
        if (!is_s_shared((int32_t) il)) {
            read_row(s_l[il], seed->state_bytes[il]);
            continue;
        }

        ggml_tensor * hand_state = d_l[il];
        if (!private_planner && (snapshot == 0 || uses_native_child_state(source_seq))) {
            read_row(hand_state, seed->state_bytes[il]);
            continue;
        }

        ggml_tensor * brain_state = s_l[il];
        if (brain_state->type != GGML_TYPE_F32 ||
            hand_state->type != GGML_TYPE_F32 ||
            brain_state->ne[0] != hand_state->ne[0]) {
            return nullptr;
        }
        const size_t n = (size_t) brain_state->ne[0];
        const size_t brain_row_size =
            ggml_row_size(brain_state->type, brain_state->ne[0]);
        const size_t hand_row_size =
            ggml_row_size(hand_state->type, hand_state->ne[0]);
        std::vector<float> public_brain(n);
        std::vector<float> private_brain(n);
        std::vector<float> planner_hand(n);
        std::vector<float> child_hand(n);
        ggml_backend_tensor_get(
            brain_state,
            public_brain.data(),
            (size_t) brain_row * brain_row_size,
            brain_row_size);
        ggml_backend_tensor_get(
            brain_state,
            private_brain.data(),
            (private_planner
                ? (size_t) n_brain_rows + (size_t) brain_row
                : 2 * (size_t) n_brain_rows + (snapshot - 1) * (size_t) n_brain_rows + (size_t) brain_row) * brain_row_size,
            brain_row_size);
        ggml_backend_tensor_get(
            hand_state,
            planner_hand.data(),
            local_row * hand_row_size,
            hand_row_size);
        for (size_t i = 0; i < n; ++i) {
            child_hand[i] = private_brain[i] + planner_hand[i] - public_brain[i];
        }
        seed->state_bytes[il].resize(hand_row_size);
        std::memcpy(
            seed->state_bytes[il].data(),
            child_hand.data(),
            hand_row_size);
    }
    if (backend_sched) {
        ggml_backend_sched_synchronize(backend_sched);
    }

    return seed;
}

bool llama_memory_recurrent::apply_hand_seed(
        llama_seq_id dest_seq,
        const std::shared_ptr<hand_seed> & seed) {
    if (!seed || seed->source_pos < 0 || dest_seq < 0 ||
        (size_t) dest_seq >= tails.size()) {
        return false;
    }

    // Validate the complete checkpoint before allocating a cell or writing
    // any layer. SEE3 contains every existing conv/local-state/F32 hand row.
    if (seed->conv_tail_bytes.size() != r_l.size() ||
        seed->state_bytes.size() != s_l.size()) {
        return false;
    }
    std::vector<std::pair<ggml_tensor *, const std::vector<uint8_t> *>> writes;
    auto stage = [&](ggml_tensor * tensor, const std::vector<uint8_t> & bytes) {
        if (!tensor) {
            return bytes.empty();
        }
        if (bytes.size() != ggml_row_size(tensor->type, tensor->ne[0])) {
            return false;
        }
        writes.emplace_back(tensor, &bytes);
        return true;
    };
    for (size_t il = 0; il < r_l.size(); ++il) {
        if (!stage(r_l[il], seed->conv_tail_bytes[il])) {
            return false;
        }
    }
    for (size_t il = 0; il < s_l.size(); ++il) {
        if (!stage(is_s_shared((int32_t) il) ? d_l[il] : s_l[il], seed->state_bytes[il])) {
            return false;
        }
    }

    const int32_t old_row = tails[(size_t) dest_seq];
    if (old_row >= 0 && ((size_t) old_row >= cells.size() ||
        !cells[(size_t) old_row].has_seq_id(dest_seq))) {
        return false;
    }
    const bool needs_cell = old_row < 0 || cells[(size_t) old_row].seq_id.size() > 1;
    int32_t row = needs_cell ? -1 : old_row;
    if (needs_cell) {
        for (uint32_t i = 0; i < size; ++i) {
            const uint32_t candidate = (head + i) % size;
            if (!cells[candidate].is_empty()) {
                continue;
            }
            row = (int32_t) candidate;
            break;
        }
    }
    if (row < 0) {
        return false;
    }
    auto write_row = [&](ggml_tensor * tensor, const std::vector<uint8_t> & bytes) {
        const size_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);
        if (bytes.size() != row_size) {
            return false;
        }
        ggml_backend_t backend = backend_sched
            ? ggml_backend_sched_get_tensor_backend(backend_sched, tensor)
            : nullptr;
        if (backend) {
            ggml_backend_tensor_set_async(
                backend,
                tensor,
                bytes.data(),
                (size_t) row * row_size,
                row_size);
        } else {
            ggml_backend_tensor_set(
                tensor,
                bytes.data(),
                (size_t) row * row_size,
                row_size);
        }
        return true;
    };

    for (const auto & write : writes) {
        GGML_ASSERT(write_row(write.first, *write.second));
    }
    if (backend_sched) {
        ggml_backend_sched_synchronize(backend_sched);
    }

    if (needs_cell) {
        if (old_row >= 0) {
            cells[(size_t) old_row].seq_id.erase(dest_seq);
        }
        cells[(size_t) row].seq_id.insert(dest_seq);
        tails[(size_t) dest_seq] = row;
        ++used;
        head = ((uint32_t) row + 1) % size;
    }
    auto & cell = cells[(size_t) row];
    cell.pos = seed->source_pos;
    cell.src = row;
    cell.src0 = row;
    set_rs_idx(dest_seq, 0);
    return true;
}

size_t llama_memory_recurrent::rerot_hand_seed_size(llama_seq_id source_seq) const {
    if (source_seq < 0 || (size_t) source_seq >= tails.size() ||
        tails[(size_t) source_seq] < 0) {
        return 0;
    }

    size_t total_bytes =
        sizeof(uint32_t) * 3 + sizeof(llama_pos);
    for (size_t il = 0; il < r_l.size(); ++il) {
        total_bytes += sizeof(uint32_t);
        if (r_l[il]) {
            total_bytes += ggml_row_size(r_l[il]->type, r_l[il]->ne[0]);
        }
    }
    for (size_t il = 0; il < s_l.size(); ++il) {
        total_bytes += sizeof(uint32_t);
        ggml_tensor * hand_state =
            is_s_shared((int32_t) il) ? d_l[il] : s_l[il];
        if (hand_state) {
            total_bytes +=
                ggml_row_size(hand_state->type, hand_state->ne[0]);
        }
    }
    return total_bytes;
}

bool llama_memory_recurrent::rerot_capture_hand_seed(llama_seq_id source_seq, std::vector<uint8_t> & seed_out) {
    seed_out.clear();
    auto seed = capture_hand_seed(0, source_seq);
    if (!seed) {
        return false;
    }

    const uint32_t magic = 0x33454553; // 'SEE3': F32 persistent hand, no implicit SEE2 conversion
    const uint32_t n_conv = uint32_t(seed->conv_tail_bytes.size());
    const uint32_t n_state = uint32_t(seed->state_bytes.size());

    size_t total_bytes =
        sizeof(magic) + sizeof(seed->source_pos) + sizeof(n_conv) + sizeof(n_state);
    for (const auto & b : seed->conv_tail_bytes) {
        total_bytes += sizeof(uint32_t) + b.size();
    }
    for (const auto & b : seed->state_bytes) {
        total_bytes += sizeof(uint32_t) + b.size();
    }

    seed_out.reserve(total_bytes);
    auto append_u32 = [&](uint32_t v) {
        const uint8_t * p = reinterpret_cast<const uint8_t *>(&v);
        seed_out.insert(seed_out.end(), p, p + sizeof(uint32_t));
    };

    append_u32(magic);
    const uint8_t * pos_bytes =
        reinterpret_cast<const uint8_t *>(&seed->source_pos);
    seed_out.insert(
        seed_out.end(),
        pos_bytes,
        pos_bytes + sizeof(seed->source_pos));
    append_u32(n_conv);
    for (const auto & b : seed->conv_tail_bytes) {
        append_u32(uint32_t(b.size()));
        seed_out.insert(seed_out.end(), b.begin(), b.end());
    }
    append_u32(n_state);
    for (const auto & b : seed->state_bytes) {
        append_u32(uint32_t(b.size()));
        seed_out.insert(seed_out.end(), b.begin(), b.end());
    }

    return true;
}

bool llama_memory_recurrent::rerot_apply_hand_seed(llama_seq_id dest_seq, const std::vector<uint8_t> & seed_in) {
    if (seed_in.size() <
        sizeof(uint32_t) * 3 + sizeof(llama_pos)) {
        return false;
    }

    size_t offset = 0;
    auto read_u32 = [&](uint32_t & v) -> bool {
        if (offset + sizeof(uint32_t) > seed_in.size()) return false;
        std::memcpy(&v, seed_in.data() + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        return true;
    };

    uint32_t magic = 0;
    if (!read_u32(magic) || magic != 0x33454553) {
        return false;
    }

    auto seed = std::make_shared<hand_seed>();
    if (offset + sizeof(seed->source_pos) > seed_in.size()) {
        return false;
    }
    std::memcpy(
        &seed->source_pos,
        seed_in.data() + offset,
        sizeof(seed->source_pos));
    offset += sizeof(seed->source_pos);

    uint32_t n_conv = 0;
    if (!read_u32(n_conv) || n_conv != r_l.size()) return false;

    seed->conv_tail_bytes.resize(n_conv);
    for (uint32_t i = 0; i < n_conv; ++i) {
        uint32_t len = 0;
        if (!read_u32(len)) return false;
        if (offset + len > seed_in.size()) return false;
        seed->conv_tail_bytes[i].assign(seed_in.data() + offset, seed_in.data() + offset + len);
        offset += len;
    }

    uint32_t n_state = 0;
    if (!read_u32(n_state) || n_state != s_l.size()) return false;

    seed->state_bytes.resize(n_state);
    for (uint32_t i = 0; i < n_state; ++i) {
        uint32_t len = 0;
        if (!read_u32(len)) return false;
        if (offset + len > seed_in.size()) return false;
        seed->state_bytes[i].assign(
            seed_in.data() + offset,
            seed_in.data() + offset + len);
        offset += len;
    }

    return offset == seed_in.size() && apply_hand_seed(dest_seq, seed);
}

bool llama_memory_recurrent::rerot_commit_rbb_frontier(
        uint32_t person_id,
        const llama_seq_id * candidate_seqs,
        const uint8_t * is_public_write,
        size_t n_candidates) {
    GGML_UNUSED(person_id);
    GGML_UNUSED(candidate_seqs);
    GGML_UNUSED(is_public_write);

    // Shared-S candidates are reduced and committed by the compute graph,
    // before llama_decode() completes. The server frontier hook remains as an
    // atomicity check but performs no host readback or backend synchronization.
    return n_candidates > 0;
}

uint32_t llama_memory_recurrent::get_recurrent_seq_used(llama_seq_id seq_id) const {
    return seq_id >= 0 && (size_t) seq_id < tails.size() && tails[(size_t) seq_id] >= 0 ? 1 : 0;
}

size_t llama_memory_recurrent::total_size() const {
    size_t size = 0;
    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_memory_recurrent::size_r_bytes() const {
    size_t size_r_bytes = 0;

    for (const auto & r : r_l) {
        if (r != nullptr) {
            size_r_bytes += ggml_nbytes(r);
        }
    }

    return size_r_bytes;
}

size_t llama_memory_recurrent::size_s_bytes() const {
    size_t size_s_bytes = 0;

    for (const auto & s : s_l) {
        if (s != nullptr) {
            size_s_bytes += ggml_nbytes(s);
        }
    }

    return size_s_bytes;
}

size_t llama_memory_recurrent::size_p_bytes() const {
    size_t size_p_bytes = 0;

    for (const auto & p : p_l) {
        if (p != nullptr) {
            size_p_bytes += ggml_nbytes(p);
        }
    }

    return size_p_bytes;
}

size_t llama_memory_recurrent::size_d_bytes() const {
    size_t size_d_bytes = 0;

    for (const auto & d : d_l) {
        if (d != nullptr) {
            size_d_bytes += ggml_nbytes(d);
        }
    }

    return size_d_bytes;
}
void llama_memory_recurrent::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);

    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges; // ranges, from inclusive, to exclusive
    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges_data; // logical source row ranges
    std::vector<int32_t> brain_rows_data; // resolved shared-brain row per serialized cell
    uint32_t cell_count = 0;

    // Count the number of cells with the specified seq_id
    // Find all the ranges of cells with this seq id (or all, when -1)
    uint32_t cell_range_begin = size;
    for (uint32_t i = 0; i < size; ++i) {
        const auto & cell = cells[i];
        if ((seq_id == -1 && !cell.is_empty()) || cell.has_seq_id(seq_id)) {
            ++cell_count;
            uint32_t rs_idx_cur = 0;

            if (n_rs_seq != 0) {
                if (seq_id != -1) {
                    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < rs_idx.size());
                    rs_idx_cur = rs_idx[seq_id];
                } else {
                    bool has_rs_idx = false;
                    for (const llama_seq_id cell_seq_id : cell.seq_id) {
                        GGML_ASSERT(cell_seq_id >= 0 && (size_t) cell_seq_id < rs_idx.size());

                        const uint32_t seq_rs_idx = rs_idx[cell_seq_id];
                        if (!has_rs_idx) {
                            rs_idx_cur = seq_rs_idx;
                            has_rs_idx = true;
                        } else if (rs_idx_cur != seq_rs_idx) {
                            GGML_ABORT("cannot write shared recurrent state with different rollback indices");
                        }
                    }
                }
            }

            const uint32_t cell_id = rs_idx_cur * size + (cell.src >= 0 ? cell.src : (int32_t) i);
            if (cell_ranges_data.empty() || cell_ranges_data.back().second != cell_id) {
                cell_ranges_data.emplace_back(cell_id, cell_id + 1);
            } else {
                cell_ranges_data.back().second++;
            }

            if (is_grouped_layout()) {
                int32_t brain_row = -1;
                const auto add_seq_brain = [&](llama_seq_id state_seq) {
                    const int32_t row = brain_read_row_for_seq(state_seq);
                    if (row < 0) {
                        throw std::runtime_error("cannot serialize grouped recurrent state without a brain row");
                    }
                    if (brain_row >= 0 && brain_row != row) {
                        throw std::runtime_error("cannot serialize one recurrent cell with different brain rows");
                    }
                    brain_row = row;
                };
                if (seq_id != -1) {
                    add_seq_brain(seq_id);
                } else {
                    for (const llama_seq_id cell_seq_id : cell.seq_id) {
                        add_seq_brain(cell_seq_id);
                    }
                }
                if (brain_row < 0) {
                    throw std::runtime_error("cannot serialize grouped recurrent state without a sequence");
                }
                brain_rows_data.push_back(brain_row);
            }

            if (cell_range_begin == size) {
                cell_range_begin = i;
            }
        } else {
            if (cell_range_begin != size) {
                cell_ranges.emplace_back(cell_range_begin, i);
                cell_range_begin = size;
            }
        }
    }
    if (cell_range_begin != size) {
        cell_ranges.emplace_back(cell_range_begin, size);
    }

    if ((flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) && cell_ranges.size() > 1) {
        GGML_ABORT("cannot save/load multiple ranges of cells to/from device memory\n");
    }

    // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
    uint32_t cell_count_check = 0;
    for (const auto & range : cell_ranges) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    cell_count_check = 0;
    for (const auto & range : cell_ranges_data) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);
    GGML_ASSERT(!is_grouped_layout() || brain_rows_data.size() == cell_count);

    io.write(&cell_count, sizeof(cell_count));

    state_write_meta(io, cell_ranges, seq_id);
    state_write_data(io, cell_ranges_data, brain_rows_data);
}

void llama_memory_recurrent::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    uint32_t cell_count;
    io.read(&cell_count, sizeof(cell_count));

    bool res = true;

    res = res && state_read_meta(io, cell_count, seq_id);

    try {
        res = res && state_read_data(io, cell_count, seq_id);
    } catch (...) {
        res = false;
    }

    if (!res) {
        if (seq_id == -1) {
            clear(true);
        } else {
            seq_rm(seq_id, -1, -1);
        }
        throw std::runtime_error("failed to restore kv cache");
    }

    if (n_rs_seq != 0) {
        if (seq_id == -1) {
            std::fill(rs_idx.begin(), rs_idx.end(), 0);
        } else {
            set_rs_idx(seq_id, 0);
        }
    }
}

void llama_memory_recurrent::state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id) const {
    for (const auto & range : cell_ranges) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            const auto & cell = cells[i];
            const llama_pos pos      = cell.pos;
            const uint32_t  n_seq_id = seq_id == -1 ? cell.seq_id.size() : 0;

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id) {
                for (auto seq_id : cell.seq_id) {
                    io.write(&seq_id, sizeof(seq_id));
                }
            }
        }
    }
}

void llama_memory_recurrent::state_write_data(
        llama_io_write_i & io,
        const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges,
        const std::vector<int32_t> & brain_rows) const {
    const uint32_t s_trans = 0;
    const uint32_t n_layer = hparams.n_layer();

    io.write(&s_trans, sizeof(s_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the R tensors first, each row is a cell
    // Get whole range at a time
    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
        if (r_l[il] == nullptr) continue;

        // Write R tensor type
        const int32_t r_type_i = (int32_t)r_l[il]->type;
        io.write(&r_type_i, sizeof(r_type_i));

        // Write row size of R tensor
        const uint64_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
        io.write(&r_size_row, sizeof(r_size_row));

        // Write each logical cell row range. With pending recurrent rollback,
        // the logical current state may live in a rollback snapshot plane.
        for (const auto & range : cell_ranges) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * r_size_row;
            io.write_tensor(r_l[il], range.first * r_size_row, buf_size);
        }
    }

    if (!s_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
            if (s_l[il] == nullptr) continue;

            // Write S tensor type
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            io.write(&s_type_i, sizeof(s_type_i));

            // Write row size of S tensor
            const uint64_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            io.write(&s_size_row, sizeof(s_size_row));

            if (is_s_shared((int32_t) il)) {
                // Grouped shared S is indexed by brain, not by the physical
                // hand row used by R/private S. Save the resolved current
                // brain twice so restore can install the ordinary public and
                // ready-private mirrors without a host round trip.
                for (int mirror = 0; mirror < 2; ++mirror) {
                    for (const int32_t brain_row : brain_rows) {
                        if (brain_row < 0 || brain_row >= s_l[il]->ne[1]) {
                            throw std::runtime_error("grouped recurrent brain row is out of bounds");
                        }
                        io.write_tensor(
                            s_l[il],
                            (size_t) brain_row * s_size_row,
                            s_size_row);
                    }
                }
            } else {
                // Write each logical cell row range. With pending recurrent
                // rollback, the current state may live in a snapshot plane.
                for (const auto & range : cell_ranges) {
                    const size_t range_size = range.second - range.first;
                    const size_t buf_size = range_size * s_size_row;
                    io.write_tensor(s_l[il], range.first * s_size_row, buf_size);
                }
            }
        }
    } else {
        // When S tensor is transposed, we also need the element size and get the element ranges from each row
        const uint32_t mem_size = size;
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
            if (s_l[il] == nullptr) continue;

            const uint32_t n_embd_s = hparams.n_embd_s();

            // Write S tensor type
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            io.write(&s_type_i, sizeof(s_type_i));

            // Write element size
            const uint32_t s_size_el = ggml_type_size(s_l[il]->type);
            io.write(&s_size_el, sizeof(s_size_el));

            // Write GQA embedding size
            io.write(&n_embd_s, sizeof(n_embd_s));

            // For each row, we get the element values of each logical cell
            for (uint32_t j = 0; j < n_embd_s; ++j) {
                for (const auto & range : cell_ranges) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * mem_size) * s_size_el;
                    const size_t buf_size = range_size * s_size_el;
                    io.write_tensor(s_l[il], src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_memory_recurrent::state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id) {
    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        if (cell_count == 0) {
            return true;
        }

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 0) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            ubatch.pos[i] = pos;
        }
        ubatch.n_seq_id[0] = 1;
        ubatch.seq_id[0] = &dest_seq_id;

        if (!find_slot(ubatch)) {
            LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
            return false;
        }

        // DEBUG CHECK: kv.head should be our first cell, kv.head + cell_count - 1 should be our last cell (verify seq_id and pos values)
        // Assume that this is one contiguous block of cells
        GGML_ASSERT(head + cell_count <= size);
        GGML_ASSERT(cells[head].pos == ubatch.pos[0]);
        GGML_ASSERT(cells[head + cell_count - 1].pos == ubatch.pos[cell_count - 1]);
        GGML_ASSERT(cells[head].has_seq_id(dest_seq_id));
        GGML_ASSERT(cells[head + cell_count - 1].has_seq_id(dest_seq_id));
    } else {
        // whole KV cache restore

        if (cell_count > size) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            auto & cell = cells[i];

            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cell.pos = pos;

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= LLAMA_MAX_SEQ) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, (unsigned) LLAMA_MAX_SEQ);
                    return false;
                }

                cell.seq_id.insert(seq_id);

                int32_t & tail = tails[(size_t) seq_id];
                if (tail != -1) {
                    LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tail);
                    return false;
                }
                tail = i;
            }
        }

        head = 0;
        used = cell_count;
    }

    for (uint32_t i = 0; i < cell_count; ++i) {
        uint32_t cell_id = head + i;
        // make sure the recurrent states will keep their restored state
        cells[cell_id].src = cell_id;
    }

    return true;
}

bool llama_memory_recurrent::state_read_data(
        llama_io_read_i & io,
        uint32_t cell_count,
        llama_seq_id dest_seq_id) {
    uint32_t s_trans;
    uint32_t n_layer;
    io.read(&s_trans, sizeof(s_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != hparams.n_layer()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, hparams.n_layer());
        return false;
    }
    if (cell_count > size) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, size);
        return false;
    }
    if (false != (bool) s_trans) {
        LLAMA_LOG_ERROR("%s: incompatible s transposition\n", __func__);
        return false;
    }

    std::vector<int32_t> brain_rows;
    if (is_grouped_layout()) {
        brain_rows.reserve(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            auto & cell = cells[head + i];
            int32_t brain_row = -1;
            for (const llama_seq_id cell_seq_id : cell.seq_id) {
                const int32_t row = brain_row_for_seq(cell_seq_id);
                if (row < 0) {
                    LLAMA_LOG_ERROR("%s: no brain row for restored seq %d\n", __func__, cell_seq_id);
                    return false;
                }
                if (brain_row >= 0 && brain_row != row) {
                    LLAMA_LOG_ERROR("%s: restored recurrent cell has different brain rows\n", __func__);
                    return false;
                }
                brain_row = row;
            }
            if (brain_row < 0 || brain_row >= (int32_t) n_brain_rows) {
                LLAMA_LOG_ERROR("%s: invalid restored brain row %d\n", __func__, brain_row);
                return false;
            }
            for (const llama_seq_id cell_seq_id : cell.seq_id) {
                seq_brain[(size_t) cell_seq_id] = brain_row;
            }
            brain_rows.push_back(brain_row);
            clear_hand_row((int32_t) (head + i));
        }
        GGML_ASSERT(dest_seq_id == -1 || brain_rows.size() <= 1);
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers
        if (r_l[il] == nullptr) continue;

        // Read type of key
        int32_t r_type_i_ref;
        io.read(&r_type_i_ref, sizeof(r_type_i_ref));
        const int32_t r_type_i = (int32_t) r_l[il]->type;
        if (r_type_i != r_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched r type (%d != %d, layer %d)\n", __func__, r_type_i, r_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t r_size_row_ref;
        io.read(&r_size_row_ref, sizeof(r_size_row_ref));
        const size_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
        if (r_size_row != r_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched r row size (%zu != %zu, layer %d)\n", __func__, r_size_row, (size_t) r_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            // Read and set the keys for the whole cell range
            io.read_tensor(r_l[il], head * r_size_row, cell_count * r_size_row);
        }
    }

    if (!s_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers
            if (s_l[il] == nullptr) continue;

            // Read type of value
            int32_t s_type_i_ref;
            io.read(&s_type_i_ref, sizeof(s_type_i_ref));
            const int32_t s_type_i = (int32_t)s_l[il]->type;

            if (s_type_i != s_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s type (%d != %d, layer %d)\n", __func__, s_type_i, s_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t s_size_row_ref;
            io.read(&s_size_row_ref, sizeof(s_size_row_ref));
            const size_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            if (s_size_row != s_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s row size (%zu != %zu, layer %d)\n", __func__, s_size_row, (size_t) s_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                if (is_s_shared((int32_t) il)) {
                    GGML_ASSERT(brain_rows.size() == cell_count);
                    for (const int32_t brain_row : brain_rows) {
                        io.read_tensor(
                            s_l[il],
                            (size_t) brain_row * s_size_row,
                            s_size_row);
                    }
                    for (const int32_t brain_row : brain_rows) {
                        io.read_tensor(
                            s_l[il],
                            ((size_t) n_brain_rows + (size_t) brain_row) * s_size_row,
                            s_size_row);
                    }
                } else {
                    // Read and set the values for the whole cell range.
                    io.read_tensor(s_l[il], head * s_size_row, cell_count * s_size_row);
                }
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers
            if (s_l[il] == nullptr) continue;

            const uint32_t n_embd_s = hparams.n_embd_s();

            // Read type of value
            int32_t s_type_i_ref;
            io.read(&s_type_i_ref, sizeof(s_type_i_ref));
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            if (s_type_i != s_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s type (%d != %d, layer %d)\n", __func__, s_type_i, s_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t s_size_el_ref;
            io.read(&s_size_el_ref, sizeof(s_size_el_ref));
            const size_t s_size_el = ggml_type_size(s_l[il]->type);
            if (s_size_el != s_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s element size (%zu != %zu, layer %d)\n", __func__, s_size_el, (size_t) s_size_el_ref, il);
                return false;
            }

            // Read state embedding size
            uint32_t n_embd_s_ref;
            io.read(&n_embd_s_ref, sizeof(n_embd_s_ref));
            if (n_embd_s != n_embd_s_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s embedding size (%u != %u, layer %d)\n", __func__, n_embd_s, n_embd_s_ref, il);
                return false;
            }

            if (cell_count) {
                // For each row in the transposed matrix, read the values for the whole cell range
                for (uint32_t j = 0; j < n_embd_s; ++j) {
                    const size_t dst_offset = (head + j * size) * s_size_el;
                    io.read_tensor(s_l[il], dst_offset, cell_count * s_size_el);
                }
            }
        }
    }

    return true;
}

//
// llama_memory_recurrent_context
//

llama_memory_recurrent_context::llama_memory_recurrent_context(llama_memory_status status) : status(status) {}

llama_memory_recurrent_context::llama_memory_recurrent_context(
        llama_memory_recurrent * mem) : status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem), is_full(true) {
}

llama_memory_recurrent_context::llama_memory_recurrent_context(
        llama_memory_recurrent * mem,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem), ubatches(std::move(ubatches)) {}

llama_memory_recurrent_context::~llama_memory_recurrent_context() = default;

bool llama_memory_recurrent_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_recurrent_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is an update
    if (ubatches.empty()) {
        // recurrent cache never performs updates
        assert(status == LLAMA_MEMORY_STATUS_NO_UPDATE);

        return true;
    }

    mem->find_slot(ubatches[i_next]);

    return true;
}

bool llama_memory_recurrent_context::postcompute_success() {
    return true;
}

llama_memory_status llama_memory_recurrent_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_recurrent_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

uint32_t llama_memory_recurrent_context::get_n_rs() const {
    return is_full ? mem->size : mem->n;
}

uint32_t llama_memory_recurrent_context::get_head() const {
    return is_full ? 0 : mem->head;
}

int32_t llama_memory_recurrent_context::get_rs_z() const {
    return is_full ? 0 : mem->rs_z;
}

uint32_t llama_memory_recurrent_context::get_size() const {
    return mem->size;
}

uint32_t llama_memory_recurrent_context::get_brain_size() const {
    return mem->is_grouped_layout() ? 2 * mem->n_brain_rows : mem->size;
}

bool llama_memory_recurrent_context::is_grouped() const {
    return mem->is_grouped_layout();
}

ggml_tensor * llama_memory_recurrent_context::get_r_l(int32_t il) const {
    return mem->r_l[il];
}

ggml_tensor * llama_memory_recurrent_context::get_s_l(int32_t il) const {
    return mem->s_l[il];
}

ggml_tensor * llama_memory_recurrent_context::get_p_l(int32_t il) const {
    return mem->p_l[il];
}

ggml_tensor * llama_memory_recurrent_context::get_d_l(int32_t il) const {
    return mem->d_l[il];
}

bool llama_memory_recurrent_context::is_s_shared(int32_t il) const {
    return mem->is_s_shared(il);
}

int32_t llama_memory_recurrent_context::brain_copy(int i) const {
    if (!mem->is_grouped_layout() || is_full) {
        return s_copy(i);
    }

    const auto & ubatch = get_ubatch();
    if (i < 0 || (uint32_t) i >= ubatch.n_seqs) {
        return -1;
    }
    const uint32_t token_index = (uint32_t) i * ubatch.n_seq_tokens;
    return mem->brain_read_row_for_seq(ubatch.seq_id[token_index][0]);
}

bool llama_memory_recurrent_context::is_child_row(int i) const {
    if (!mem->is_grouped_layout() || is_full) return false;
    const auto & ubatch = get_ubatch();
    if (i < 0 || (uint32_t) i >= ubatch.n_seqs) return false;
    const llama_seq_id seq = ubatch.seq_id[(uint32_t) i * ubatch.n_seq_tokens][0];
    return seq >= 0 && (size_t) seq < mem->seq_episode.size() && mem->seq_episode[(size_t) seq] != 0 &&
        mem->seq_node[(size_t) seq] != 0 && mem->seq_node[(size_t) seq] != LLAMA_REROT_NODE_INVALID;
}

bool llama_memory_recurrent_context::is_public_write(int i) const {
    if (!mem->is_grouped_layout() || is_full) {
        return false;
    }
    const auto & ubatch = get_ubatch();
    if (i < 0 || (uint32_t) i >= ubatch.n_seqs) {
        return false;
    }
    const uint32_t token_index = (uint32_t) i * ubatch.n_seq_tokens;
    const llama_seq_id seq_id = ubatch.seq_id[token_index][0];
    if (seq_id < 0 || (size_t) seq_id >= mem->seq_public_write.size()) {
        return false;
    }

    // Before an episode write tag exists this is ordinary causal recurrence
    // (including the root prompt prefill), so its single writer must commit.
    return mem->seq_episode[(size_t) seq_id] == 0 ||
        mem->seq_public_write[(size_t) seq_id] != 0;
}

std::map<int32_t, std::vector<int32_t>>
llama_memory_recurrent_context::public_brain_groups() const {
    std::map<int32_t, std::vector<int32_t>> result;
    if (!mem->is_grouped_layout() || is_full) {
        return result;
    }
    const auto & ubatch = get_ubatch();
    for (uint32_t i = 0; i < ubatch.n_seqs; ++i) {
        const uint32_t token_index = i * ubatch.n_seq_tokens;
        const llama_seq_id seq_id = ubatch.seq_id[token_index][0];
        const int32_t base_row = mem->brain_row_for_seq(seq_id);
        if (base_row < 0) {
            continue;
        }

        if (mem->seq_episode[(size_t) seq_id] == 0) {
            // Keep a ready private planner snapshot without a D2H/H2D copy at
            // episode start.
            result[base_row].push_back((int32_t) i);
            result[(int32_t) mem->n_brain_rows + base_row].push_back((int32_t) i);
        } else if (mem->seq_public_write[(size_t) seq_id] != 0) {
            // Every public writer belongs to the person's shared frontier,
            // including child workers. Restricting this to node zero silently
            // turned all parallel lanes into isolated recurrent decoders.
            result[base_row].push_back((int32_t) i);
        } else if (mem->seq_node[(size_t) seq_id] == 0) {
            result[(int32_t) mem->n_brain_rows + base_row].push_back((int32_t) i);
        }
    }
    return result;
}

int32_t llama_memory_recurrent_context::s_copy(int i) const {
    const uint32_t cell_idx = i + mem->head;
    const int32_t  src0     = mem->cells[cell_idx].src0;

    if (mem->n_rs_seq == 0) {
        return src0;
    }

    uint32_t idx = 0;
    if (!mem->cells[cell_idx].seq_id.empty()) {
        const llama_seq_id seq = *mem->cells[cell_idx].seq_id.begin();
        if (seq >= 0 && (size_t) seq < mem->rs_idx.size()) {
            idx = mem->rs_idx[seq];
            // reset rollback idx
            mem->rs_idx[seq] = 0;
        }
    }
    return (int32_t)(idx * mem->size) + src0;
}
