#include "testing.h"

#include "llama.h"

#include "../src/llama-batch.h"
#include "../src/llama-memory.h"
#include "../src/llama-vocab.h"

#include <cstdlib>
#include <initializer_list>
#include <map>
#include <string>
#include <utility>
#include <vector>

// mock memory that only provides per-sequence position ranges
struct mock_memory : public llama_memory_i {
    std::map<llama_seq_id, std::pair<llama_pos, llama_pos>> ranges; // seq_id -> [pos_min, pos_max]

    llama_memory_context_ptr init_batch(llama_batch_allocr &, uint32_t, bool) override {  GGML_ASSERT(false && "not implemented"); }
    llama_memory_context_ptr init_full() override {  GGML_ASSERT(false && "not implemented"); }
    llama_memory_context_ptr init_update(llama_context *, bool) override { GGML_ASSERT(false && "not implemented"); }

    bool get_can_shift() const override { GGML_ASSERT(false && "not implemented"); }

    void clear(bool) override { GGML_ASSERT(false && "not implemented"); }

    bool seq_rm  (llama_seq_id, llama_pos, llama_pos) override { GGML_ASSERT(false && "not implemented"); }
    void seq_cp  (llama_seq_id, llama_seq_id, llama_pos, llama_pos) override { GGML_ASSERT(false && "not implemented"); }
    void seq_keep(llama_seq_id) override { GGML_ASSERT(false && "not implemented"); }
    void seq_add (llama_seq_id, llama_pos, llama_pos, llama_pos) override { GGML_ASSERT(false && "not implemented"); }
    void seq_div (llama_seq_id, llama_pos, llama_pos, int) override { GGML_ASSERT(false && "not implemented");  }

    llama_pos seq_pos_min(llama_seq_id seq_id) const override {
        auto it = ranges.find(seq_id);
        return it == ranges.end() ? -1 : it->second.first;
    }

    llama_pos seq_pos_max(llama_seq_id seq_id) const override {
        auto it = ranges.find(seq_id);
        return it == ranges.end() ? -1 : it->second.second;
    }

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override { return {}; }

    void state_write(llama_io_write_i &, llama_seq_id, llama_state_seq_flags) const override { GGML_ASSERT(false && "not implemented"); }
    void state_read (llama_io_read_i &,  llama_seq_id, llama_state_seq_flags) override { GGML_ASSERT(false && "not implemented"); }
};

// builds embedding batches - an empty llama_vocab rejects all token ids, so
// the tests use embeddings everywhere except the token validation tests
struct batch_builder {
    uint32_t n_embd;

    std::vector<float>     embd;
    std::vector<llama_pos> pos;
    std::vector<int32_t>   n_seq_id;
    std::vector<int8_t>    logits;

    std::vector<std::vector<llama_seq_id>> seq;
    std::vector<llama_seq_id *>            seq_ptr;

    batch_builder(uint32_t n_embd = 2) : n_embd(n_embd) {}

    // embd values are 100*i + k so that ubatch contents can be traced back to batch indices
    void add(llama_pos p, std::initializer_list<llama_seq_id> seq_ids, bool output) {
        const int32_t i = (int32_t) seq.size();
        for (uint32_t k = 0; k < n_embd; ++k) {
            embd.push_back(100.0f*i + k);
        }
        pos.push_back(p);
        n_seq_id.push_back((int32_t) seq_ids.size());
        seq.emplace_back(seq_ids);
        logits.push_back(output ? 1 : 0);
    }

    llama_batch make(bool with_pos = true, bool with_seq = true, bool with_logits = true) {
        seq_ptr.clear();
        for (auto & s : seq) {
            seq_ptr.push_back(s.data());
        }
        seq_ptr.push_back(nullptr);

        llama_batch res = {};
        res.n_tokens = (int32_t) seq.size();
        res.embd     = embd.data();
        res.pos      = with_pos    ? pos.data()      : nullptr;
        res.n_seq_id = with_seq    ? n_seq_id.data() : nullptr;
        res.seq_id   = with_seq    ? seq_ptr.data()  : nullptr;
        res.logits   = with_logits ? logits.data()   : nullptr;

        return res;
    }
};

static void test_init(testing & t) {
    llama_vocab vocab;

    t.test("rejects_n_seq_max_too_large", [&](testing & t) {
        batch_builder bb;
        bb.add(0, {0}, true);

        llama_batch_allocr ba(1);
        t.assert_true(!ba.init(bb.make(), vocab, nullptr, bb.n_embd, LLAMA_MAX_SEQ + 1, false));
    });

    t.test("rejects_invalid_token", [&](testing & t) {
        llama_token tok = 0; // empty vocab -> every token id is out of range
        llama_batch batch = llama_batch_get_one(&tok, 1);

        llama_batch_allocr ba(1);
        t.assert_true("token id >= n_tokens", !ba.init(batch, vocab, nullptr, 0, 1, false));

        tok = -1;
        t.assert_true("negative token id", !ba.init(batch, vocab, nullptr, 0, 1, false));
    });

    t.test("rejects_invalid_seq_id", [&](testing & t) {
        llama_batch_allocr ba(1);

        {
            batch_builder bb;
            bb.add(0, {4}, true);
            t.assert_true("seq_id >= n_seq_max", !ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));
        }
        {
            batch_builder bb;
            bb.add(0, {-1}, true);
            t.assert_true("negative seq_id", !ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));
        }
    });

    t.test("autofill_defaults", [&](testing & t) {
        batch_builder bb;
        for (int i = 0; i < 4; ++i) {
            bb.add(0, {0}, false);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(false, false, false), vocab, nullptr, bb.n_embd, 4, false));

        const llama_batch & batch = ba.get_batch();

        t.assert_equal(4u, ba.get_n_tokens());

        for (int i = 0; i < 4; ++i) {
            t.assert_equal("pos defaults to 0..n-1", i, batch.pos[i]);
            t.assert_equal("n_seq_id defaults to 1", 1, batch.n_seq_id[i]);
            t.assert_equal("seq_id defaults to 0",   0, batch.seq_id[i][0]);
        }

        t.assert_equal("only the last token is an output", 1u, ba.get_n_outputs());
        t.assert_equal(0, (int) batch.logits[0]);
        t.assert_equal(1, (int) batch.logits[3]);

        t.assert_equal(0, ba.seq_pos_min(0));
        t.assert_equal(3, ba.seq_pos_max(0));
        t.assert_equal(-1, ba.seq_pos_min(1));
    });

    t.test("output_all", [&](testing & t) {
        batch_builder bb;
        for (int i = 0; i < 4; ++i) {
            bb.add(i, {0}, false);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(true, true, false), vocab, nullptr, bb.n_embd, 4, true));
        t.assert_equal(4u, ba.get_n_outputs());
    });

    t.test("explicit_logits", [&](testing & t) {
        batch_builder bb;
        bb.add(0, {0}, true);
        bb.add(1, {0}, false);
        bb.add(2, {0}, true);

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));
        t.assert_equal(2u, ba.get_n_outputs());

        llama_ubatch ub = ba.split_simple(10);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(1, (int) ub.output[0]);
        t.assert_equal(0, (int) ub.output[1]);
        t.assert_equal(1, (int) ub.output[2]);

        const auto & out_ids = ba.get_out_ids();
        t.assert_equal((size_t) 2, out_ids.size());
        t.assert_equal(0, out_ids[0]);
        t.assert_equal(2, out_ids[1]);
    });

    t.test("pos_from_memory", [&](testing & t) {
        mock_memory mem;
        mem.ranges[0] = {0, 9};

        batch_builder bb;
        for (int i = 0; i < 3; ++i) {
            bb.add(0, {0}, false);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(false, true, false), vocab, &mem, bb.n_embd, 4, false));

        t.assert_equal("pos continues after memory", 10, ba.seq_pos_min(0));
        t.assert_equal(12, ba.seq_pos_max(0));
    });

    t.test("pos_continuity_with_memory", [&](testing & t) {
        mock_memory mem;
        mem.ranges[0] = {0, 9};

        llama_batch_allocr ba(1);

        {
            batch_builder bb;
            bb.add(10, {0}, false);
            bb.add(11, {0}, true);
            t.assert_true("pos_max + 1 is accepted", ba.init(bb.make(), vocab, &mem, bb.n_embd, 4, false));
        }
        {
            batch_builder bb;
            bb.add(11, {0}, false);
            bb.add(12, {0}, true);
            t.assert_true("gap after memory is rejected", !ba.init(bb.make(), vocab, &mem, bb.n_embd, 4, false));
        }
        {
            batch_builder bb;
            bb.add(9, {0}, false);
            bb.add(10, {0}, true);
            t.assert_true("overlap with memory is rejected", !ba.init(bb.make(), vocab, &mem, bb.n_embd, 4, false));
        }
    });

    t.test("rejects_non_continuous_positions", [&](testing & t) {
        batch_builder bb;
        bb.add(0, {0}, false);
        bb.add(1, {0}, false);
        bb.add(3, {0}, true);

        llama_batch_allocr ba(1);
        t.assert_true(!ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));
    });

    t.test("rejects_decreasing_positions", [&](testing & t) {
        batch_builder bb;
        const llama_pos    pos[7] = {4, 5, 0, 1, 6, 2, 3};
        const llama_seq_id seq[7] = {0, 0, 1, 1, 0, 1, 0};
        for (int i = 0; i < 7; ++i) {
            bb.add(pos[i], {seq[i]}, false);
        }
        // seq 0 sees positions 4,5,6,3 in batch order -> the trailing 3 decreases

        llama_batch_allocr ba(1);
        t.assert_true(!ba.init(bb.make(true, true, false), vocab, nullptr, bb.n_embd, 4, false));
    });

    t.test("allows_equal_positions_in_seq", [&](testing & t) {
        batch_builder bb;
        bb.add(0, {0}, false);
        bb.add(0, {0}, false);
        bb.add(1, {0}, true);

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(true, true, false), vocab, nullptr, bb.n_embd, 4, false));
    });


    t.test("rejects_coupled_diverged_seqs", [&](testing & t) {
        batch_builder bb;
        bb.add(6, {0, 1}, true);

        llama_batch_allocr ba(1);

        mock_memory mem;
        mem.ranges[0] = {0, 5};
        mem.ranges[1] = {2, 5}; // same pos_max, different pos_min -> diverged
        t.assert_true(!ba.init(bb.make(), vocab, &mem, bb.n_embd, 4, false));

        mem.ranges[1] = {0, 5};
        t.assert_true(ba.init(bb.make(), vocab, &mem, bb.n_embd, 4, false));
    });
}

static void test_split(testing & t) {
    llama_vocab vocab;

    t.test("split_simple_chunks", [&](testing & t) {
        batch_builder bb;
        for (int i = 0; i < 5; ++i) {
            bb.add(i, {0}, i == 4);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));

        llama_ubatch ub = ba.split_simple(2);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_true(!ub.equal_seqs());
        t.assert_equal(1u, ub.n_seqs_unq);
        t.assert_equal(0, ub.seq_id_unq[0]);
        t.assert_equal(0, ub.seq_idx[0]);
        for (int i = 0; i < 2; ++i) {
            t.assert_equal(i, ub.pos[i]);
            t.assert_equal(1, ub.n_seq_id[i]);
            t.assert_equal(0, ub.seq_id[i][0]);
            t.assert_equal(100.0f*i, ub.embd[i*bb.n_embd]);
            t.assert_equal(100.0f*i + 1, ub.embd[i*bb.n_embd + 1]);
        }

        ub = ba.split_simple(2);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_equal(2, ub.pos[0]);
        t.assert_equal(3, ub.pos[1]);

        ub = ba.split_simple(2);
        t.assert_equal(1u, ub.n_tokens);
        t.assert_equal(4, ub.pos[0]);
        t.assert_equal(1, (int) ub.output[0]);

        t.assert_equal(5u, ba.get_n_used());

        ub = ba.split_simple(2);
        t.assert_equal("batch is consumed", 0u, ub.n_tokens);

        const auto & out_ids = ba.get_out_ids();
        t.assert_equal((size_t) 1, out_ids.size());
        t.assert_equal(4, out_ids[0]);
    });

    t.test("split_reset_allows_resplit", [&](testing & t) {
        batch_builder bb;
        for (int i = 0; i < 3; ++i) {
            bb.add(i, {0}, i == 2);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));

        while (ba.split_simple(1).n_tokens > 0) {
        }
        t.assert_equal(3u, ba.get_n_used());

        ba.split_reset();
        t.assert_equal(0u, ba.get_n_used());

        llama_ubatch ub = ba.split_simple(10);
        t.assert_equal(3u, ub.n_tokens);
    });

    t.test("split_equal_unequal_lengths", [&](testing & t) {
        batch_builder bb;
        for (int i = 0; i < 4; ++i) {
            bb.add(i, {0}, i == 3);
        }
        for (int i = 0; i < 2; ++i) {
            bb.add(i, {1}, i == 1);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));

        llama_ubatch ub = ba.split_equal(8, false, 0);
        t.assert_true(ub.equal_seqs());
        t.assert_equal("both seqs advance by the shorter length", 4u, ub.n_tokens);
        t.assert_equal(2u, ub.n_seq_tokens);
        t.assert_equal(2u, ub.n_seqs);
        t.assert_equal(2u, ub.n_seqs_unq);
        // tokens are grouped per sequence set: [s0 s0 s1 s1]
        t.assert_equal(0, ub.seq_id[0][0]);
        t.assert_equal(0, ub.seq_id[1][0]);
        t.assert_equal(1, ub.seq_id[2][0]);
        t.assert_equal(1, ub.seq_id[3][0]);
        t.assert_equal(0, ub.pos[0]);
        t.assert_equal(1, ub.pos[1]);
        t.assert_equal(0, ub.pos[2]);
        t.assert_equal(1, ub.pos[3]);

        ub = ba.split_equal(8, false, 0);
        t.assert_equal("only seq 0 remains", 2u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(2, ub.pos[0]);
        t.assert_equal(3, ub.pos[1]);

        ub = ba.split_equal(8, false, 0);
        t.assert_equal(0u, ub.n_tokens);

        t.assert_equal(6u, ba.get_n_used());
    });

    t.test("split_equal_coupled", [&](testing & t) {
        batch_builder bb;
        bb.add(0, {0, 1}, false);
        bb.add(1, {0, 1}, true);

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));

        llama_ubatch ub = ba.split_equal(4, true, 0);
        t.assert_equal("sequential split rejects coupled seqs", 0u, ub.n_tokens);

        ub = ba.split_equal(4, false, 0);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_equal("one sequence set", 1u, ub.n_seqs);
        t.assert_equal("two unique seq ids", 2u, ub.n_seqs_unq);
        t.assert_equal(2, ub.n_seq_id[0]);
        t.assert_equal(0, ub.seq_idx[0]);
        t.assert_equal(1, ub.seq_idx[1]);
    });

    t.test("split_seq_per_sequence", [&](testing & t) {
        batch_builder bb;
        for (llama_seq_id s = 0; s < 3; ++s) {
            bb.add(0, {s}, false);
            bb.add(1, {s}, true);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));

        for (llama_seq_id s = 0; s < 3; ++s) {
            llama_ubatch ub = ba.split_seq(8);
            t.assert_equal(2u, ub.n_tokens);
            t.assert_equal(1u, ub.n_seqs);
            t.assert_equal(s, ub.seq_id[0][0]);
            t.assert_equal(s, ub.seq_id_unq[0]);
        }

        t.assert_equal(0u, ba.split_seq(8).n_tokens);
        t.assert_equal(6u, ba.get_n_used());
    });

    t.test("ubatch_reserve", [&](testing & t) {
        llama_batch_allocr ba(1);

        llama_ubatch ub = ba.ubatch_reserve(3, 2);
        t.assert_equal(6u, ub.n_tokens);
        t.assert_equal(3u, ub.n_seq_tokens);
        t.assert_equal(2u, ub.n_seqs);
        t.assert_equal(2u, ub.n_seqs_unq);
        t.assert_true(ub.equal_seqs());
        t.assert_equal(0, ub.seq_id_unq[0]);
        t.assert_equal(1, ub.seq_id_unq[1]);
        t.assert_true(ub.token != nullptr);
        t.assert_true(ub.embd == nullptr);
    });
}

static void test_keep_tail(testing & t) {
    llama_vocab vocab;

    // batch with n_tokens[s] tokens for each seq s, output on the last token of each seq
    auto make_batch = [](batch_builder & bb, std::initializer_list<int> n_tokens) {
        llama_seq_id s = 0;
        for (int n : n_tokens) {
            for (int i = 0; i < n; ++i) {
                bb.add(i, {s}, i == n - 1);
            }
            ++s;
        }
        return bb.make();
    };

    t.test("noop_when_seqs_complete", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {2, 2}), vocab, nullptr, bb.n_embd, 4, false));

        llama_ubatch ub = ba.split_equal(4, false, 2);
        t.assert_equal("both seqs fit whole", 4u, ub.n_tokens);
        t.assert_equal(2u, ub.n_seqs);
        t.assert_equal(2u, ub.n_seq_tokens);

        t.assert_equal(0u, ba.split_equal(4, false, 2).n_tokens);
    });

    t.test("defers_seq_with_short_remainder", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {2, 3}), vocab, nullptr, bb.n_embd, 4, false));

        // expansion stops at 2 tokens per seq: seq 0 completes, seq 1 would be left
        // with 1 < n_keep_tail remaining, so it is deferred entirely
        llama_ubatch ub = ba.split_equal(4, true, 2);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(0, ub.seq_id[0][0]);
        t.assert_equal(2u, ba.get_n_used());

        ub = ba.split_equal(4, true, 2);
        t.assert_equal("deferred seq comes back whole", 3u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(1, ub.seq_id[0][0]);
        for (int i = 0; i < 3; ++i) {
            t.assert_equal(i, ub.pos[i]);
        }

        t.assert_equal(5u, ba.get_n_used());
        t.assert_equal(0u, ba.split_equal(4, true, 2).n_tokens);
    });

    t.test("completes_first_seq_when_all_violate", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {3, 3}), vocab, nullptr, bb.n_embd, 4, false));

        // expansion stops at 2 tokens per seq, leaving both with 1 < n_keep_tail remaining;
        // seq 0 still fits in n_ubatch, so it is extended to completion and emitted alone
        llama_ubatch ub = ba.split_equal(4, false, 2);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(3u, ub.n_seq_tokens);
        t.assert_equal(0, ub.seq_id[0][0]);
        for (int i = 0; i < 3; ++i) {
            t.assert_equal(i, ub.pos[i]);
        }
        t.assert_equal(3u, ba.get_n_used());

        ub = ba.split_equal(4, false, 2);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(1, ub.seq_id[0][0]);
        t.assert_equal(6u, ba.get_n_used());
    });

    t.test("truncates_to_preserve_tail", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {5}), vocab, nullptr, bb.n_embd, 4, false));

        // 4 tokens would leave a remainder of 1, and the seq does not fit in n_ubatch,
        // so the ubatch is truncated until n_keep_tail tokens remain
        llama_ubatch ub = ba.split_equal(4, false, 2);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(2, ub.pos[2]);
        t.assert_equal(3u, ba.get_n_used());

        ub = ba.split_equal(4, false, 2);
        t.assert_equal("trailing tokens stay in one ubatch", 2u, ub.n_tokens);
        t.assert_equal(3, ub.pos[0]);
        t.assert_equal(4, ub.pos[1]);
        t.assert_equal(1, (int) ub.output[1]);

        t.assert_equal(5u, ba.get_n_used());
    });

    t.test("keeps_full_ubatch_with_sufficient_remainder", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {6}), vocab, nullptr, bb.n_embd, 4, false));

        llama_ubatch ub = ba.split_equal(4, false, 2);
        t.assert_equal("remainder >= n_keep_tail, no truncation", 4u, ub.n_tokens);

        ub = ba.split_equal(4, false, 2);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_equal(4, ub.pos[0]);
        t.assert_equal(5, ub.pos[1]);

        t.assert_equal(6u, ba.get_n_used());
    });

    t.test("multi_seq_prefix_kept", [&](testing & t) {
        batch_builder bb;

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(make_batch(bb, {3, 4}), vocab, nullptr, bb.n_embd, 6, false));

        // expansion stops at 3 tokens per seq: seq 0 completes, seq 1 has 1 < n_keep_tail
        // remaining and is deferred even though its tokens were already gathered
        llama_ubatch ub = ba.split_equal(6, true, 2);
        t.assert_equal(3u, ub.n_tokens);
        t.assert_equal(1u, ub.n_seqs);
        t.assert_equal(0, ub.seq_id[0][0]);
        t.assert_equal(3u, ba.get_n_used());

        ub = ba.split_equal(6, true, 2);
        t.assert_equal(4u, ub.n_tokens);
        t.assert_equal(1, ub.seq_id[0][0]);
        t.assert_equal(7u, ba.get_n_used());
    });
}

static void test_mrope(testing & t) {
    llama_vocab vocab;

    t.test("pos_layout_and_split", [&](testing & t) {
        const uint32_t n_pos = 4;
        const uint32_t n_embd = 2;

        batch_builder bb(n_embd);
        bb.add(10, {0}, false);
        bb.add(11, {0}, true);

        // M-RoPE positions for embeddings are laid out [n_pos][n_tokens]
        std::vector<llama_pos> pos = {
            10, 11, // temporal
             5,  6, // y
             7,  8, // x
             0,  0,
        };

        llama_batch batch = bb.make(false, true, true);
        batch.pos = pos.data();

        llama_batch_allocr ba(n_pos);
        t.assert_true(ba.init(batch, vocab, nullptr, n_embd, 4, false));

        llama_ubatch ub = ba.split_simple(2);
        t.assert_equal(2u, ub.n_tokens);
        t.assert_equal(n_pos, ub.n_pos);
        t.assert_true(ub.is_pos_2d());

        const llama_pos expected[8] = {10, 11, 5, 6, 7, 8, 0, 0};
        for (int i = 0; i < 8; ++i) {
            t.assert_equal(expected[i], ub.pos[i]);
        }
    });

    t.test("pos_jump_allowed", [&](testing & t) {
        const uint32_t n_pos = 4;
        const uint32_t n_embd = 2;

        mock_memory mem;
        mem.ranges[0] = {0, 9};

        llama_batch_allocr ba(n_pos);

        auto try_pos = [&](llama_pos p0) {
            batch_builder bb(n_embd);
            bb.add(p0, {0}, true);

            std::vector<llama_pos> pos = {p0, 1, 1, 0};

            llama_batch batch = bb.make(false, true, true);
            batch.pos = pos.data();

            return ba.init(batch, vocab, &mem, n_embd, 4, false);
        };

        t.assert_true("gap after memory is allowed",     try_pos(15));
        t.assert_true("overlap is allowed for embd",     try_pos(9));
        t.assert_true("pos behind memory is rejected",  !try_pos(8));
    });
}

static void set_pool_flag(const char * value) {
#ifdef _WIN32
    _putenv_s("TP5_UBATCH_POOL", value ? value : "");
#else
    if (value) {
        setenv("TP5_UBATCH_POOL", value, 1);
    } else {
        unsetenv("TP5_UBATCH_POOL");
    }
#endif
}

static bool same_ubatch(const llama_ubatch & a, const llama_ubatch & b, uint32_t n_embd) {
    if (a.n_tokens     != b.n_tokens     ||
        a.n_seq_tokens != b.n_seq_tokens ||
        a.n_seqs       != b.n_seqs       ||
        a.n_seqs_unq   != b.n_seqs_unq   ||
        a.n_pos        != b.n_pos        ||
        a.equal_seqs() != b.equal_seqs() ||
        (a.token == nullptr) != (b.token == nullptr) ||
        (a.embd  == nullptr) != (b.embd  == nullptr)) {
        return false;
    }

    for (uint32_t i = 0; i < a.n_tokens; ++i) {
        if (a.n_seq_id[i] != b.n_seq_id[i] || a.output[i] != b.output[i]) {
            return false;
        }
        for (int s = 0; s < a.n_seq_id[i]; ++s) {
            if (a.seq_id[i][s] != b.seq_id[i][s]) {
                return false;
            }
        }
        if (a.token && a.token[i] != b.token[i]) {
            return false;
        }
        if (a.embd) {
            for (uint32_t k = 0; k < n_embd; ++k) {
                if (a.embd[i*n_embd + k] != b.embd[i*n_embd + k]) {
                    return false;
                }
            }
        }
    }

    for (uint32_t i = 0; i < a.n_tokens*a.n_pos; ++i) {
        if (a.pos[i] != b.pos[i]) {
            return false;
        }
    }

    for (uint32_t i = 0; i < LLAMA_MAX_SEQ; ++i) {
        if (a.seq_idx[i] != b.seq_idx[i]) {
            return false;
        }
    }

    for (uint32_t i = 0; i < a.n_seqs_unq; ++i) {
        if (a.seq_id_unq[i] != b.seq_id_unq[i]) {
            return false;
        }
    }

    return true;
}

static void test_host_pool(testing & t) {
    llama_vocab vocab;

    t.test("split_values_match_reference", [&](testing & t) {
        batch_builder bb_ref;
        batch_builder bb_pool;

        const auto add_all = [](batch_builder & bb) {
            bb.add(0, {0, 1}, false);
            bb.add(0, {0}, false);
            bb.add(1, {0}, false);
            bb.add(1, {1}, true);
            bb.add(2, {0}, false);
        };

        add_all(bb_ref);
        add_all(bb_pool);

        set_pool_flag(nullptr);
        llama_batch_allocr ref(1);
        t.assert_true(ref.init(bb_ref.make(), vocab, nullptr, bb_ref.n_embd, 4, false));

        set_pool_flag("1");
        llama_batch_allocr pooled(1);
        t.assert_true(pooled.init(bb_pool.make(), vocab, nullptr, bb_pool.n_embd, 4, false));

        int splits = 0;
        while (true) {
            llama_ubatch a = ref.split_simple(2);
            llama_ubatch b = pooled.split_simple(2);

            t.assert_equal("same n_tokens", a.n_tokens, b.n_tokens);
            if (a.n_tokens == 0) {
                break;
            }

            t.assert_true("pooled handle", b.data.pooled());
            t.assert_true("split values match the reference", same_ubatch(a, b, bb_ref.n_embd));
            ++splits;
        }

        t.assert_equal("three splits", 3, splits);
        t.assert_equal("both batches consumed", ref.get_n_used(), pooled.get_n_used());
    });

    t.test("reused_slot_clears_absent_seq_idx", [&](testing & t) {
        set_pool_flag("1");

        batch_builder bb;
        bb.add(0, {0, 1}, false);
        bb.add(0, {0}, false);
        bb.add(1, {0}, false);
        bb.add(1, {0}, false);

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));

        uint32_t gen = 0;
        {
            llama_ubatch ub = ba.split_simple(1);
            t.assert_equal("first ubatch has both seqs", 1, ub.seq_idx[1]);
            gen = ub.data.generation();
        }

        llama_ubatch ub = ba.split_simple(1);
        t.assert_true("reused slot", ub.data.pooled());
        t.assert_equal("generation advances", gen + 1, ub.data.generation());
        t.assert_equal("absent seq idx resets to -1", -1, ub.seq_idx[1]);
        t.assert_equal("present seq idx is set", 0, ub.seq_idx[0]);
    });

    t.test("exhaustion_falls_back", [&](testing & t) {
        set_pool_flag("1");

        batch_builder bb;
        for (int i = 0; i < 4; ++i) {
            bb.add(i, {0}, false);
        }

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));

        std::vector<llama_ubatch> held;
        for (int i = 0; i < 4; ++i) {
            llama_ubatch ub = ba.split_simple(1);
            t.assert_equal(1u, ub.n_tokens);
            held.push_back(std::move(ub));
        }

        int pooled = 0;
        for (const auto & ub : held) {
            pooled += ub.data.pooled() ? 1 : 0;
        }
        t.assert_equal("capacity n_ubatch + 2 holds three", 3, pooled);

        held.clear();

        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));
        llama_ubatch ub = ba.split_simple(1);
        t.assert_true("slot is pooled again after release", ub.data.pooled());
    });

    t.test("reserve_is_heap_owned_and_zeroed", [&](testing & t) {
        set_pool_flag("1");

        llama_batch_allocr ba(1);

        {
            llama_ubatch ub = ba.ubatch_reserve(2, 2);
            // A reserve ubatch can outlive the allocator, so it must never
            // borrow a pool slot.
            t.assert_true("reserve payload is heap-owned", !ub.data.pooled());

            for (uint32_t i = 0; i < ub.n_tokens; ++i) {
                ub.token[i]    = 7;
                ub.output[i]   = 1;
                ub.n_seq_id[i] = 3;
            }
            for (uint32_t i = 0; i < ub.n_tokens*ub.n_pos; ++i) {
                ub.pos[i] = 9;
            }
        }

        llama_ubatch ub = ba.ubatch_reserve(2, 2);
        t.assert_true("second reserve is heap-owned", !ub.data.pooled());

        for (uint32_t i = 0; i < ub.n_tokens; ++i) {
            t.assert_equal("token is cleared", 0, ub.token[i]);
            t.assert_equal("output is cleared", 0, (int) ub.output[i]);
            t.assert_equal("n_seq_id is cleared", 0, ub.n_seq_id[i]);
        }
        for (uint32_t i = 0; i < ub.n_tokens*ub.n_pos; ++i) {
            t.assert_equal("pos is cleared", 0, ub.pos[i]);
        }
        t.assert_true("decision order is cleared", ub.decision_order == nullptr);
        t.assert_true("embd stays empty", ub.embd == nullptr);

        set_pool_flag(nullptr);
        llama_batch_allocr ref(1);
        llama_ubatch r = ref.ubatch_reserve(2, 2);
        t.assert_true("flag off uses the heap", !r.data.pooled());
        t.assert_equal("heap reserve token is zero", 0, r.token[0]);
        t.assert_equal("heap reserve output is zero", 0, (int) r.output[0]);
    });

    t.test("reserve_outlives_allocr", [&](testing & t) {
        set_pool_flag("1");

        llama_ubatch kept = {};
        {
            llama_batch_allocr ba(1);
            kept = ba.ubatch_reserve(2, 2);
            t.assert_true("reserve payload does not borrow the pool", !kept.data.pooled());
        }

        // The allocator and its pool are gone. The heap payload must still be
        // readable and must release without touching a destroyed mutex.
        t.assert_equal("payload survives the allocator", 4u, kept.n_tokens);
        t.assert_equal("token survives", 0, kept.token[0]);
        t.assert_equal("seq idx survives", 0, kept.seq_idx[0]);
        t.assert_equal("second seq idx survives", 1, kept.seq_idx[1]);
        t.assert_equal("unique seq id survives", 1, kept.seq_id_unq[1]);

        set_pool_flag(nullptr);
    });

    t.test("handle_outlives_variable", [&](testing & t) {
        set_pool_flag("1");

        batch_builder bb;
        bb.add(0, {0}, false);
        bb.add(1, {0}, true);

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));

        llama_ubatch ub = ba.split_simple(2);
        llama_ubatch copy = ub;
        ub = llama_ubatch{};

        t.assert_true("copy keeps the payload", copy.data.pooled());
        t.assert_equal(2u, copy.n_tokens);
        t.assert_equal(0, copy.pos[0]);
        t.assert_equal(1, copy.pos[1]);

        set_pool_flag(nullptr);
    });

    t.test("graph_copy_pins_slot", [&](testing & t) {
        set_pool_flag("1");

        batch_builder bb1;
        bb1.add(0, {0}, false);
        bb1.add(1, {0}, true);

        llama_batch_allocr ba(1);
        t.assert_true(ba.init(bb1.make(), vocab, nullptr, bb1.n_embd, 4, false));

        llama_ubatch first = ba.split_simple(2);
        t.assert_true("first split is pooled", first.data.pooled());
        const uint32_t slot = first.data.slot();

        // llm_graph_params keeps an intentional copy of the ubatch.
        llama_ubatch graph = first;
        first = llama_ubatch{};

        batch_builder bb2;
        bb2.add(10, {0}, false);
        bb2.add(11, {0}, true);

        t.assert_true(ba.init(bb2.make(), vocab, nullptr, bb2.n_embd, 4, false));
        llama_ubatch second = ba.split_simple(2);

        t.assert_true("second split is pooled", second.data.pooled());
        t.assert_true("pinned slot is not reused", second.data.slot() != slot);
        t.assert_equal("graph copy keeps the old position", 0, graph.pos[0]);
        t.assert_equal("graph copy keeps the old tail", 1, graph.pos[1]);
        t.assert_equal("second split has the new position", 10, second.pos[0]);

        // Dropping the last copy returns the slot.
        graph = llama_ubatch{};

        batch_builder bb3;
        bb3.add(20, {0}, false);
        bb3.add(21, {0}, true);

        t.assert_true(ba.init(bb3.make(), vocab, nullptr, bb3.n_embd, 4, false));
        llama_ubatch third = ba.split_simple(2);
        t.assert_equal("released slot is reused", slot, third.data.slot());
        t.assert_equal("third split has the new position", 20, third.pos[0]);

        set_pool_flag(nullptr);
    });

    t.test("split_outlives_allocr", [&](testing & t) {
        set_pool_flag("1");

        // The exact graph_reserve lifetime: the allocator dies, the graph
        // keeps a ubatch copy, and the copy is read afterwards.
        llama_ubatch graph = {};
        {
            batch_builder bb;
            bb.add(3, {0}, false);
            bb.add(4, {0}, true);

            llama_batch_allocr ba(1);
            t.assert_true(ba.init(bb.make(), vocab, nullptr, bb.n_embd, 4, false));

            llama_ubatch ub = ba.split_simple(2);
            t.assert_true("split is pooled", ub.data.pooled());
            graph = ub;
        }

        t.assert_true("handle keeps the pool alive", graph.data.pooled());
        t.assert_equal("token count survives", 2u, graph.n_tokens);
        t.assert_equal("position survives", 3, graph.pos[0]);
        t.assert_equal("tail position survives", 4, graph.pos[1]);
        t.assert_equal("seq id survives", 0, graph.seq_id[0][0]);
        t.assert_equal("output survives", 1, (int) graph.output[1]);
        t.assert_equal("seq idx survives", 0, graph.seq_idx[0]);

        graph = llama_ubatch{};
        t.assert_true("release after the allocator died is safe", true);

        set_pool_flag(nullptr);
    });
}

int main(int argc, char ** argv) {
    testing t;

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }
    if (!t.verbose) {
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    }

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("init",      test_init);
    t.test("split",     test_split);
    t.test("keep_tail", test_keep_tail);
    t.test("mrope",     test_mrope);
    t.test("host_pool", test_host_pool);

    return t.summary();
}
