// verify/host_mask_boundary.cpp — CPU boundary check for the production KQ-mask fill.
//
// NOT EXECUTED in this session (user discipline: code + static review only).
// Exact build/run commands: ../RUNBOOK.md.
//
// Includes the production translation unit so the real template
// set_input_kq_mask_impl<T, causal, ...> is the code under test. No production
// file is modified. Widths 1/255/256/257/511/512, single-stream and
// two-stream/three-sequence (multi-column) layouts with holes, F16 and F32,
// causal and non-causal, canaries on both ends of every row block.
//
// WITH_NKV_TRUE_LEN is defined only for the baseline tree (stock + frozen
// 6e8fb4bb patch); the stock tree lacks seq_used_max_p1 and the last case is
// skipped there. All other output must be byte-identical between the two trees
// (OFF == stock checks compare the stdout of both builds).

#include "llama-kv-cache.cpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void expect(bool ok, const std::string & what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) {
        ++g_fail;
    }
}

struct row_spec {
    llama_seq_id seq;
    llama_pos   pos;
};

// reference model for a token row: keep cell j iff it exists in the token's
// stream, carries the token's sequence and (causal) its position is not after
// the token's position. Holes and foreign sequences are masked.
bool ref_keep(const std::vector<llama_kv_cells> & cells, uint32_t stream, uint32_t j,
              llama_seq_id seq, llama_pos p1, bool causal) {
    const auto & c = cells.at(stream);
    if (j >= c.size()) {
        return false;
    }
    if (c.is_empty(j)) {
        return false;
    }
    if (!c.seq_has(j, seq)) {
        return false;
    }
    if (causal && c.pos_get(j) > p1) {
        return false;
    }
    return true;
}

// every sequence's first token in the batch must start at or before its lowest
// used cell position + 32, else the production per-sequence row-reuse
// optimization (seq_idxs) would legitimately skip cells and a full-row
// reference model would no longer apply. Data below satisfies it by
// construction; the check documents why the reference is exact.
bool reference_applicable(const std::vector<llama_kv_cells> & cells,
                          const std::vector<uint32_t> & seq_to_stream,
                          const std::vector<row_spec> & rows, uint32_t n_stream) {
    const uint32_t n_tps = (uint32_t) (rows.size() / n_stream);
    (void) n_tps;
    for (const auto & r : rows) {
        llama_pos first = INT32_MAX;
        for (const auto & o : rows) {
            if (o.seq == r.seq) {
                first = std::min(first, o.pos);
            }
        }
        const auto & c = cells.at(seq_to_stream.at(r.seq));
        for (uint32_t j = 0; j < c.size(); ++j) {
            if (c.is_empty(j) || !c.seq_has(j, r.seq)) {
                continue;
            }
            if (c.pos_get(j) < first - 32) {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
void fill_and_check(const std::string & label,
                    const std::vector<llama_kv_cells> & cells,
                    const std::vector<uint32_t> & seq_to_stream,
                    const std::vector<row_spec> & rows,
                    uint32_t n_stream, int64_t width, bool causal,
                    uint32_t keep_bits, uint32_t drop_bits) {
    llama_hparams hp{};

    std::vector<llama_pos>       pos(rows.size());
    std::vector<llama_seq_id>    seq_store(rows.size());
    std::vector<llama_seq_id *>  seq_ptrs(rows.size());
    std::vector<int32_t>         n_seq_id(rows.size(), 1);
    for (size_t i = 0; i < rows.size(); ++i) {
        pos[i]       = rows[i].pos;
        seq_store[i] = rows[i].seq;
        seq_ptrs[i]  = &seq_store[i];
    }

    llama_ubatch ubatch{};
    ubatch.n_tokens   = (uint32_t) rows.size();
    ubatch.n_pos      = 1;
    ubatch.n_seqs     = 1;
    ubatch.n_seqs_unq = n_stream;
    ubatch.pos        = pos.data();
    ubatch.seq_id     = seq_ptrs.data();
    ubatch.n_seq_id   = n_seq_id.data();

    const int64_t n_tps = (int64_t) (rows.size() / n_stream);

    const args_set_input_kq_mask args = {
        /*.hparams       =*/ hp,
        /*.ubatch        =*/ &ubatch,
        /*.v_cells       =*/ cells,
        /*.seq_to_stream =*/ seq_to_stream,
        /*.n_swa         =*/ 0,
        /*.swa_type      =*/ LLAMA_SWA_TYPE_NONE,
        /*.n_kv          =*/ width,
        /*.n_stream      =*/ (int64_t) n_stream,
        /*.n_tps         =*/ n_tps,
    };

    std::vector<T> buf(rows.size() * (size_t) width + 2);
    // explicit canary bit patterns: 0x1234 (binary16) / 12345.0f (binary32)
    for (size_t ci = 0; ci < buf.size(); ++ci) {
        if (sizeof(T) == 2) {
            const uint16_t c = 0x1234;
            std::memcpy(&buf[ci], &c, 2);
        } else {
            const float c = 12345.0f;
            std::memcpy(&buf[ci], &c, sizeof(float));
        }
    }

    if (causal) {
        set_input_kq_mask_impl<T, true,  false, false, false>(args, buf.data() + 1);
    } else {
        set_input_kq_mask_impl<T, false, false, false, false>(args, buf.data() + 1);
    }

    bool ok = true;
    std::string detail;

    auto bits_of = [](const T & v) -> uint32_t {
        uint32_t b = 0;
        if (sizeof(T) == 2) {
            uint16_t h = 0;
            std::memcpy(&h, &v, 2);
            b = h;
        } else {
            std::memcpy(&b, &v, 4);
        }
        return b;
    };

    if (sizeof(T) == 2) {
        uint16_t c = 0;
        std::memcpy(&c, &buf.front(), 2);
        ok = ok && (c == 0x1234);
        std::memcpy(&c, &buf.back(), 2);
        ok = ok && (c == 0x1234);
    } else {
        float c = 0.0f;
        std::memcpy(&c, &buf.front(), 4);
        ok = ok && (c == 12345.0f);
        std::memcpy(&c, &buf.back(), 4);
        ok = ok && (c == 12345.0f);
    }

    for (uint32_t i = 0; i < rows.size() && ok; ++i) {
        const uint32_t stream = seq_to_stream.at(rows[i].seq);
        for (int64_t j = 0; j < width; ++j) {
            const bool keep = ref_keep(cells, stream, (uint32_t) j, rows[i].seq, rows[i].pos, causal);
            const uint32_t got = bits_of(buf[1 + (size_t) i * (size_t) width + (size_t) j]);
            const uint32_t want = keep ? keep_bits : drop_bits;
            if (got != want) {
                ok = false;
                char tmp[160];
                std::snprintf(tmp, sizeof(tmp), " row=%u col=%lld got=0x%08x want=0x%08x",
                              i, (long long) j, got, want);
                detail = tmp;
                break;
            }
        }
    }

    expect(ok, label + " width=" + std::to_string(width) + " rows=" + std::to_string(rows.size()) +
               " streams=" + std::to_string(n_stream) + (causal ? " causal" : " non-causal") +
               " canaries" + detail);
}
void setup_single(std::vector<llama_kv_cells> & cells, int64_t width) {
    cells.assign(1, llama_kv_cells());
    cells[0].resize((uint32_t) width);
    cells[0].pos_set(0, 0);
    cells[0].seq_add(0, 0);
    if (width > 1) {
        cells[0].pos_set(1, 1);
        cells[0].seq_add(1, 0);
    }
    if (width > 3) {
        cells[0].pos_set((uint32_t) width - 1, (llama_pos) (width - 1));
        cells[0].seq_add((uint32_t) width - 1, 0);
    }
}

void setup_multi(std::vector<llama_kv_cells> & cells, std::vector<uint32_t> & seq_to_stream,
                 int64_t width) {
    cells.assign(2, llama_kv_cells());
    cells[0].resize((uint32_t) width);
    cells[1].resize((uint32_t) width);

    seq_to_stream.assign(LLAMA_MAX_SEQ, 0);
    seq_to_stream[0] = 0;
    seq_to_stream[1] = 0;
    seq_to_stream[2] = 1;

    // stream 0: seq 0 at cell 0; seq 1 at cells 3 and width-1
    cells[0].pos_set(0, 0);
    cells[0].seq_add(0, 0);
    cells[0].pos_set(3, 0);
    cells[0].seq_add(3, 1);
    cells[0].pos_set((uint32_t) width - 1, 1);
    cells[0].seq_add((uint32_t) width - 1, 1);

    // stream 1: seq 2 at cells 0 and width-2 (hole between)
    cells[1].pos_set(0, 0);
    cells[1].seq_add(0, 2);
    cells[1].pos_set((uint32_t) width - 2, 3);
    cells[1].seq_add((uint32_t) width - 2, 2);
}

} // namespace

int main() {
    const int64_t widths[] = { 1, 255, 256, 257, 511, 512 };

    // F16 = what the FA path allocates; F32 = non-FA path. Exact bit patterns.
    const uint32_t f16_keep = 0x0000, f16_drop = 0xfc00; // IEEE binary16 +0.0 / -inf
    const uint32_t f32_keep = 0x00000000u, f32_drop = 0xff800000u;

    for (int64_t width : widths) {
        std::vector<llama_kv_cells> cells;
        std::vector<uint32_t> seq_to_stream(LLAMA_MAX_SEQ, 0);
        std::vector<row_spec> rows = { { 0, 0 }, { 0, 1 } };

        setup_single(cells, width);
        expect(reference_applicable(cells, seq_to_stream, rows, 1),
               "reference model applicable single-stream width=" + std::to_string(width));
        fill_and_check<ggml_fp16_t>("single-stream F16", cells, seq_to_stream, rows, 1, width, true,
                                    f16_keep, f16_drop);
        fill_and_check<float>("single-stream F32", cells, seq_to_stream, rows, 1, width, true,
                              f32_keep, f32_drop);
        fill_and_check<ggml_fp16_t>("single-stream F16", cells, seq_to_stream, rows, 1, width, false,
                                    f16_keep, f16_drop);
    }

    for (int64_t width : { 256, 257, 512 }) {
        std::vector<llama_kv_cells> cells;
        std::vector<uint32_t> seq_to_stream;
        std::vector<row_spec> rows = { { 0, 0 }, { 1, 0 }, { 2, 0 }, { 2, 1 } };

        setup_multi(cells, seq_to_stream, width);
        expect(reference_applicable(cells, seq_to_stream, rows, 2),
               "reference model applicable multi-stream width=" + std::to_string(width));
        fill_and_check<ggml_fp16_t>("multi-stream F16", cells, seq_to_stream, rows, 2, width, true,
                                    f16_keep, f16_drop);
        fill_and_check<float>("multi-stream F32", cells, seq_to_stream, rows, 2, width, true,
                              f32_keep, f32_drop);
        fill_and_check<ggml_fp16_t>("multi-stream F16", cells, seq_to_stream, rows, 2, width, false,
                                    f16_keep, f16_drop);
    }

#ifdef WITH_NKV_TRUE_LEN
    // production width (true frontier) feeding the production fill: the mask is
    // allocated at exactly seq_used_max_p1(seq) columns and every live cell of
    // that sequence must stay inside the width, with the last column used.
    for (int64_t capacity : { 256, 257, 511, 512 }) {
        std::vector<llama_kv_cells> cells(1);
        cells[0].resize((uint32_t) capacity);
        cells[0].pos_set(0, 0);
        cells[0].seq_add(0, 0);
        if (capacity > 1) {
            cells[0].pos_set((uint32_t) capacity - 1, (llama_pos) (capacity - 1));
            cells[0].seq_add((uint32_t) capacity - 1, 0);
        }

        const uint32_t w = cells[0].seq_used_max_p1(0);

        bool covered = true;
        for (uint32_t j = 0; j < cells[0].size(); ++j) {
            if (!cells[0].is_empty(j) && cells[0].seq_has(j, 0) && j >= w) {
                covered = false;
            }
        }
        expect(covered, "true frontier covers every live cell seq0 capacity=" +
                        std::to_string(capacity) + " width=" + std::to_string(w));
        expect(w == capacity, "true frontier equals highest used cell + 1 capacity=" +
                              std::to_string(capacity));

        std::vector<uint32_t> seq_to_stream(LLAMA_MAX_SEQ, 0);
        // two tokens of the same sequence: positions must start at the sequence's
        // minimum used position, otherwise the production row-reuse shortcut
        // (cells with p0 < seq_min_pos - 32 stay out of seq_idxs) makes a
        // full-row reference inapplicable by design
        std::vector<row_spec> rows = { { 0, 0 }, { 0, (llama_pos) capacity - 1 } };
        expect(reference_applicable(cells, seq_to_stream, rows, 1),
               "reference model applicable true-width capacity=" + std::to_string(capacity));
        fill_and_check<ggml_fp16_t>("true-width F16", cells, seq_to_stream, rows, 1, (int64_t) w,
                                    true, f16_keep, f16_drop);
    }
#endif

    std::printf("%s host_mask_boundary failures=%d\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
