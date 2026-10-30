#include <llama.h>
#include <ggml-backend.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr llama_seq_id reader_id = 5;
constexpr llama_seq_id peer_id = 6;
constexpr llama_seq_id replacement_id = 7;
using logits = std::vector<float>;

void require(bool value, const std::string & message) {
    if (!value) { throw std::runtime_error(message); }
}

struct batch_owner {
    explicit batch_owner(int count) : value(llama_batch_init(count, 0, 1)) {}
    ~batch_owner() { llama_batch_free(value); }
    batch_owner(const batch_owner &) = delete;
    batch_owner & operator=(const batch_owner &) = delete;
    llama_batch value;
};

struct query {
    llama_seq_id seq;
    llama_pos pos;
    llama_token token;
    std::vector<llama_attention_entry> entries;
};

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool special) {
    const auto size = llama_tokenize(vocab, text.data(), int32_t(text.size()), nullptr, 0, special, false);
    require(size < 0 && size != INT32_MIN, "could not size fixture tokens");
    std::vector<llama_token> out(static_cast<std::size_t>(-size));
    const auto count = llama_tokenize(vocab, text.data(), int32_t(text.size()), out.data(), -size, special, false);
    require(count > 0 && count <= -size, "could not tokenize fixture");
    out.resize(static_cast<std::size_t>(count));
    return out;
}

class session {
public:
    session(llama_model * model, int threads, ggml_type cache_type, bool expect_unsupported = false)
        : vocabulary_(llama_model_get_vocab(model)) {
        auto params = llama_context_default_params();
        params.n_ctx = 512;
        params.n_ctx_kv = 512;
        params.n_seq_max = 8;
        params.n_rs_slots = 3;
        params.n_batch = 64;
        params.n_ubatch = 64;
        params.n_outputs_max = 64;
        params.n_threads = threads;
        params.n_threads_batch = threads;
        params.n_rs_seq = 0;
        params.kv_unified = true;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        params.type_k = cache_type;
        params.type_v = cache_type;
        params.no_perf = true;
        params.batch_invariant_matmul = true;
        context_.reset(llama_init_from_model(model, params));
        require(context_ != nullptr, "could not create the real test context");
        require(llama_get_batch_invariant_matmul(context_.get()),
                "the explicit batch-invariant matmul policy was not retained");
        const bool supported = llama_attention_views_supported(context_.get());
        require(supported != expect_unsupported, expect_unsupported ?
                "the deliberately unsupported reader profile was reported as supported" :
                "reader views are unsupported for this model and cache profile");
        require(llama_n_seq_max(context_.get()) >= 8, "logical sequence capacity was reduced");
    }

    ~session() { if (context_) { llama_synchronize(context_.get()); } }

    void prefill(const std::vector<llama_token> & tokens) {
        require(!tokens.empty() && tokens.size() <= 64, "invalid fixture prefix");
        std::vector<query> rows;
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            rows.push_back({0, llama_pos(i), tokens[i], {}});
        }
        (void) execute(rows, false, false);
        llama_memory_seq_cp(memory(), 0, reader_id, -1, -1);
        llama_memory_seq_cp(memory(), 0, peer_id, -1, -1);
        require(llama_memory_seq_pos_max(memory(), reader_id) == llama_pos(tokens.size() - 1),
                "base copy did not initialize the reader");
    }

    void expect_view_rejected(const query & row) {
        require(!row.entries.empty(), "the unsupported-profile probe needs a valid view");
        const llama_attention_view view{row.seq, row.pos, row.entries.back().virtual_pos,
                                        row.entries.size(), row.entries.data()};
        require(!llama_attention_set_views(context_.get(), &view, 1),
                "the deliberately unsupported reader profile accepted an explicit view");
    }

    std::map<llama_seq_id, logits> execute(std::vector<query> rows, bool explicit_views,
                                         bool reject_invalid) {
        require(!rows.empty() && rows.size() <= 64, "invalid fixture row count");
        if (explicit_views) {
            std::vector<llama_attention_view> views;
            for (auto & row : rows) {
                require(!row.entries.empty(), "empty fixture view");
                views.push_back({row.seq, row.pos, row.entries.back().virtual_pos,
                                 row.entries.size(), row.entries.data()});
            }
            require(llama_attention_set_views(context_.get(), views.data(), views.size()),
                    "valid reader views were rejected");
            if (reject_invalid) {
                auto invalid = rows.front().entries;
                invalid.pop_back();
                auto bad = views.front();
                bad.entries = invalid.data();
                bad.n_entries = invalid.size();
                require(!llama_attention_set_views(context_.get(), &bad, 1),
                        "view without its current key was accepted");
                ++invalid_rejections;
                invalid = rows.front().entries;
                const auto foreign = rows.front().seq == reader_id ? peer_id : reader_id;
                invalid.insert(invalid.end() - 1, {foreign, rows.front().pos, 0});
                for (std::size_t i = 0; i < invalid.size(); ++i) {
                    invalid[i].virtual_pos = llama_pos(i);
                }
                bad.query_virtual_pos = invalid.back().virtual_pos;
                bad.n_entries = invalid.size();
                bad.entries = invalid.data();
                require(!llama_attention_set_views(context_.get(), &bad, 1),
                        "uncomputed foreign current key was accepted");
                ++invalid_rejections;
                invalid = rows.front().entries;
                invalid.insert(invalid.begin() + 1, {rows.front().seq, 0, 1});
                for (std::size_t i = 0; i < invalid.size(); ++i) {
                    invalid[i].virtual_pos = llama_pos(i);
                }
                bad.query_virtual_pos = invalid.back().virtual_pos;
                bad.n_entries = invalid.size();
                bad.entries = invalid.data();
                require(!llama_attention_set_views(context_.get(), &bad, 1),
                        "one physical prefix key was accepted twice through sequence aliases");
                ++invalid_rejections;
            }
            // The setter must own its metadata. Destroy every caller view
            // before the actual operation, after the rejected replacements.
            for (auto & row : rows) { row.entries.clear(); row.entries.shrink_to_fit(); }
            views.clear();
        } else { llama_attention_clear_views(context_.get()); }
        batch_owner owner(int(rows.size()));
        owner.value.n_tokens = int32_t(rows.size());
        for (std::size_t i = 0; i < rows.size(); ++i) {
            owner.value.token[i] = rows[i].token;
            owner.value.pos[i] = rows[i].pos;
            owner.value.n_seq_id[i] = 1;
            owner.value.seq_id[i][0] = rows[i].seq;
            owner.value.logits[i] = true;
        }
        const auto status = llama_decode(context_.get(), owner.value);
        llama_synchronize(context_.get());
        require(status == 0, "actual decode failed: " + std::to_string(status));
        ++calls;
        input_rows += rows.size();
        std::map<llama_seq_id, logits> out;
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto data = llama_get_logits_ith(context_.get(), int32_t(i));
            require(data != nullptr, "actual query has no logits");
            logits values(data, data + llama_vocab_n_tokens(vocabulary_));
            for (float value : values) { require(std::isfinite(value), "non-finite real model logit"); }
            out[rows[i].seq] = std::move(values);
        }
        llama_attention_clear_views(context_.get());
        return out;
    }

    void retire_and_replace(llama_pos base, llama_token token,
                            const std::vector<llama_attention_entry> & prefix,
                            bool check_full_pool) {
        if (check_full_pool) {
            // C_base plus both live workers use all three physical rows.
            // Copying the immutable base adds a logical alias, not capacity.
            llama_memory_seq_cp(memory(), 0, replacement_id, -1, -1);
            auto entries = prefix;
            entries.push_back({replacement_id, base, llama_pos(entries.size())});
            const llama_attention_view view{replacement_id, base, entries.back().virtual_pos,
                                            entries.size(), entries.data()};
            require(llama_attention_set_views(context_.get(), &view, 1),
                    "full-pool fixture view was invalid before admission");
            batch_owner batch(1);
            batch.value.n_tokens = 1;
            batch.value.token[0] = token;
            batch.value.pos[0] = base;
            batch.value.n_seq_id[0] = 1;
            batch.value.seq_id[0][0] = replacement_id;
            batch.value.logits[0] = true;
            const auto status = llama_decode(context_.get(), batch.value);
            llama_synchronize(context_.get());
            llama_attention_clear_views(context_.get());
            ++calls;
            ++input_rows;
            require(status != 0, "a new native lineage advanced despite the full physical recurrent pool");
            ++capacity_rejections;
        }
        require(llama_memory_seq_rm_recurrent(memory(), peer_id), "recurrent retirement failed");
        // Hybrid seq_pos_max is the minimum of attention and recurrent
        // positions. It can become -1 here. Only later attention consumption
        // and unchanged reader logits prove that the peer keys survived.
        if (!check_full_pool) { llama_memory_seq_cp(memory(), 0, replacement_id, -1, -1); }
        auto entries = prefix;
        entries.push_back({replacement_id, base, llama_pos(entries.size())});
        (void) execute({{replacement_id, base, token, std::move(entries)}}, true, false);
    }

    std::vector<std::uint8_t> save_reader() const {
        const auto size = llama_state_seq_get_size(context_.get(), reader_id);
        require(size > 0 && size < std::size_t(512) * 1024 * 1024, "invalid reader snapshot size");
        std::vector<std::uint8_t> bytes(size);
        require(llama_state_seq_get_data(context_.get(), bytes.data(), bytes.size(), reader_id) == size,
                "reader snapshot failed");
        return bytes;
    }

    void restore_reader(const std::vector<std::uint8_t> & bytes) {
        require(llama_memory_seq_rm(memory(), reader_id, -1, -1), "reader removal failed");
        require(llama_state_seq_set_data(context_.get(), bytes.data(), bytes.size(), reader_id) == bytes.size(),
                "reader snapshot restore failed");
    }

    std::size_t calls = 0;
    std::size_t input_rows = 0;
    std::size_t invalid_rejections = 0;
    std::size_t capacity_rejections = 0;

private:
    llama_memory_t memory() const { return llama_get_memory(context_.get()); }
    const llama_vocab * vocabulary_;
    std::unique_ptr<llama_context, decltype(&llama_free)> context_{nullptr, llama_free};
};

struct behavior {
    bool visible = true;
    bool ordinary = false;
    bool reverse = false;
    bool serial = false;
    bool invalid = false;
    bool retire = false;
    bool restore = false;
    bool full_pool = false;
};

struct result {
    std::array<logits, 3> reader;
    logits restored;
    std::size_t calls;
    std::size_t rows;
    std::size_t rejected;
    std::size_t capacity_rejected;
};

result run_case(llama_model * model, int threads, ggml_type cache_type, const std::vector<llama_token> & base,
                const std::array<llama_token, 3> & own, llama_token first_peer,
                llama_token second_peer, behavior mode) {
    session run(model, threads, cache_type);
    run.prefill(base);
    const auto position = llama_pos(base.size());
    std::vector<llama_attention_entry> prefix;
    for (llama_pos i = 0; i < position; ++i) { prefix.push_back({0, i, i}); }
    const auto view = [&](llama_seq_id stage, int round) {
        auto entries = prefix;
        if (mode.visible && round > 0) {
            const auto peer = stage == reader_id ? peer_id : reader_id;
            for (int i = 0; i < round; ++i) {
                entries.push_back({peer, position + i, llama_pos(entries.size())});
            }
        }
        for (int i = 0; i <= round; ++i) {
            entries.push_back({stage, position + i, llama_pos(entries.size())});
        }
        return entries;
    };
    result out;
    for (int round = 0; round < 2; ++round) {
        std::vector<query> rows = {
            {peer_id, position + round, round == 0 ? first_peer : second_peer, view(peer_id, round)},
            {reader_id, position + round, own[std::size_t(round)], view(reader_id, round)},
        };
        if (mode.reverse) { std::reverse(rows.begin(), rows.end()); }
        std::map<llama_seq_id, logits> values;
        if (mode.serial) {
            // Both owned views were frozen before either operation. A prior
            // call can allocate a peer's current key but cannot expose it.
            for (auto & row : rows) {
                auto value = run.execute({row}, !mode.ordinary, false);
                values.insert(std::make_move_iterator(value.begin()), std::make_move_iterator(value.end()));
            }
        } else { values = run.execute(rows, !mode.ordinary, mode.invalid); }
        out.reader[std::size_t(round)] = std::move(values.at(reader_id));
    }
    if (mode.retire) { run.retire_and_replace(position, first_peer, prefix, mode.full_pool); }
    const auto saved = mode.restore ? run.save_reader() : std::vector<std::uint8_t>{};
    query last{reader_id, position + 2, own[2], view(reader_id, 2)};
    out.reader[2] = run.execute({last}, !mode.ordinary, false).at(reader_id);
    if (mode.restore) {
        run.restore_reader(saved);
        out.restored = run.execute({last}, !mode.ordinary, false).at(reader_id);
    }
    out.calls = run.calls;
    out.rows = run.input_rows;
    out.rejected = run.invalid_rejections;
    out.capacity_rejected = run.capacity_rejections;
    return out;
}

double max_abs(const logits & a, const logits & b) {
    require(a.size() == b.size() && !a.empty(), "logit shape mismatch");
    double maximum = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "non-finite comparison input");
        maximum = std::max(maximum, std::abs(double(a[i]) - double(b[i])));
    }
    return maximum;
}

class numerical_checks {
public:
    numerical_checks(bool keep_going, bool require_cross_shape)
        : keep_going_(keep_going), require_cross_shape_(require_cross_shape) {}

    double same(const logits & a, const logits & b, const std::string & name,
                bool cross_shape = false) {
        const double difference = max_abs(a, b);
        std::size_t failed_logits = 0;
        double maximum_ratio = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const auto limit = 2e-4 + 1e-5 * std::max(std::abs(double(a[i])), std::abs(double(b[i])));
            const auto actual = std::abs(double(a[i]) - double(b[i]));
            if (actual > limit) { ++failed_logits; }
            maximum_ratio = std::max(maximum_ratio, actual / limit);
        }
        std::ostringstream record;
        record << std::setprecision(17) << "{\"comparison\":" << std::quoted(name)
               << ",\"status\":\"" << (failed_logits == 0 ? "PASS" : "FAIL")
               << "\",\"scope\":\"" << (cross_shape ? "cross_shape" : "same_shape")
               << "\",\"required\":" << ((!cross_shape || require_cross_shape_) ? "true" : "false")
               << ",\"qualification\":\""
               << ((cross_shape && !require_cross_shape_) ? "NOT_QUALIFIED" : failed_logits == 0 ? "PASS" : "FAIL")
               << "\",\"max_abs\":" << difference << ",\"checked_logits\":" << a.size()
               << ",\"failed_logits\":" << failed_logits
               << ",\"max_tolerance_ratio\":" << maximum_ratio << '}';
        save(record.str(), failed_logits == 0, cross_shape);
        require(failed_logits == 0 || keep_going_ || (cross_shape && !require_cross_shape_),
                name + ": logits changed, max_abs=" + std::to_string(difference));
        return difference;
    }

    void effect(double difference, const std::string & name) {
        const bool passed = difference > 1e-3;
        std::ostringstream record;
        record << std::setprecision(17) << "{\"comparison\":" << std::quoted(name)
               << ",\"status\":\"" << (passed ? "PASS" : "FAIL")
               << "\",\"scope\":\"same_shape_causal_effect\",\"required\":true"
               << ",\"qualification\":\"" << (passed ? "PASS" : "FAIL")
               << "\",\"max_abs\":" << difference << ",\"required_minimum\":0.001}";
        save(record.str(), passed, false);
        require(passed || keep_going_, name + ": committed peer token produced no detectable logit effect");
    }

    std::size_t failures() const { return failures_; }

    void print(std::ostream & output) const {
        output << '[';
        for (std::size_t i = 0; i < records_.size(); ++i) {
            if (i != 0) { output << ','; }
            output << records_[i];
        }
        output << ']';
    }

    void print_qualifications(std::ostream & output) const {
        output << "{\"same_shape_attention_and_lifecycle\":{\"status\":\""
               << (same_shape_failures_ ? "FAIL" : same_shape_checks_ ? "PASS" : "NOT_QUALIFIED")
               << "\",\"comparisons\":" << same_shape_checks_
               << ",\"failed_comparisons\":" << same_shape_failures_
               << "},\"cross_shape_equivalence\":{\"status\":\""
               << (!require_cross_shape_ ? "NOT_QUALIFIED" : cross_shape_failures_ ? "FAIL" :
                   cross_shape_checks_ ? "PASS" : "NOT_QUALIFIED")
               << "\",\"explicitly_required\":" << (require_cross_shape_ ? "true" : "false")
               << ",\"observed_comparisons\":" << cross_shape_checks_
               << ",\"observed_failed_comparisons\":" << cross_shape_failures_
               << ",\"scope\":\"executed batch and serial shapes only; no general native batch-invariance claim\"}}";
    }

private:
    void save(std::string record, bool passed, bool cross_shape) {
        if (cross_shape) {
            ++cross_shape_checks_;
            if (!passed) { ++cross_shape_failures_; }
        } else {
            ++same_shape_checks_;
            if (!passed) { ++same_shape_failures_; }
        }
        if (!passed && (!cross_shape || require_cross_shape_)) { ++failures_; }
        std::cerr << record << '\n';
        records_.push_back(std::move(record));
    }
    bool keep_going_;
    bool require_cross_shape_;
    std::size_t failures_ = 0;
    std::size_t same_shape_checks_ = 0;
    std::size_t same_shape_failures_ = 0;
    std::size_t cross_shape_checks_ = 0;
    std::size_t cross_shape_failures_ = 0;
    std::vector<std::string> records_;
};

int integer(const std::string & text) {
    int value = -1;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && value >= 0,
            "invalid integer argument");
    return value;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        std::string path;
        int layers = 0;
        int threads = 2;
        ggml_type cache_type = GGML_TYPE_F32;
        bool keep_going = false;
        bool require_cross_shape = false;
        bool expect_unsupported = false;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--keep-going") {
                require(!keep_going, "--keep-going was supplied twice");
                keep_going = true;
                continue;
            }
            if (option == "--require-cross-shape-equivalence") {
                require(!require_cross_shape, "--require-cross-shape-equivalence was supplied twice");
                require_cross_shape = true;
                continue;
            }
            if (option == "--expect-unsupported") {
                require(!expect_unsupported, "--expect-unsupported was supplied twice");
                expect_unsupported = true;
                continue;
            }
            require(i + 1 < argc, "usage: test-rerot-reader-views --model MODEL [--gpu-layers N] [--threads N] [--kv-type f16|f32] [--keep-going] [--require-cross-shape-equivalence] [--expect-unsupported]");
            if (option == "--model" && path.empty()) { path = argv[++i]; }
            else if (option == "--gpu-layers") { layers = integer(argv[++i]); }
            else if (option == "--threads") { threads = integer(argv[++i]); }
            else if (option == "--kv-type") {
                const std::string value = argv[++i];
                require(value == "f16" || value == "f32", "KV type must be f16 or f32");
                cache_type = value == "f16" ? GGML_TYPE_F16 : GGML_TYPE_F32;
            }
            else { throw std::invalid_argument("unknown option: " + option); }
        }
        require(!path.empty() && threads > 0, "a real model and positive thread count are required");
        require(!expect_unsupported || cache_type == GGML_TYPE_F16,
                "--expect-unsupported tests the explicit F16 reader capability boundary; specify --kv-type f16");
        require(!expect_unsupported || !require_cross_shape,
                "an unsupported reader profile cannot qualify cross-shape reader equivalence");
        numerical_checks numeric(keep_going, require_cross_shape);
        ggml_backend_load_all();
        llama_backend_init();
        struct owner { ~owner() { llama_backend_free(); } } backend;
        auto params = llama_model_default_params();
        params.n_gpu_layers = layers;
        params.split_mode = LLAMA_SPLIT_MODE_NONE;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(path.c_str(), params), llama_model_free);
        require(model != nullptr, "could not load the real model");
        const auto vocab = llama_model_get_vocab(model.get());
        const auto base = tokenize(vocab, "The shared work begins.", true);
        const auto token = [&](const char * text) { return tokenize(vocab, text, false).back(); };
        const std::array<llama_token, 3> own{token(" alpha"), token(" beta"), token(" gamma")};
        if (expect_unsupported) {
            // The rejection probe contains only actual prefix keys and the
            // reader's own next key. Its metadata is valid independently of
            // the deliberately unsupported cache precision. A real ordinary
            // continuation after the rejected setter must still work.
            session run(model.get(), threads, cache_type, true);
            run.prefill(base);
            std::vector<llama_attention_entry> entries;
            for (llama_pos i = 0; i < llama_pos(base.size()); ++i) { entries.push_back({0, i, i}); }
            entries.push_back({reader_id, llama_pos(base.size()), llama_pos(base.size())});
            const query next{reader_id, llama_pos(base.size()), own[0], std::move(entries)};
            run.expect_view_rejected(next);
            const auto values = run.execute({next}, false, false).at(reader_id);
            require(!values.empty() && run.calls == 2, "ordinary A did not execute both sides of the capability probe");
            std::cout << "{\"status\":\"PASS\",\"scope\":\"explicit unsupported F16 reader capability boundary\""
                      << ",\"kv_type\":\"f16\",\"views_supported\":false,\"valid_view_rejected\":true"
                      << ",\"batch_invariant_matmul\":true"
                      << ",\"ordinary_continuation_logits\":" << values.size()
                      << ",\"actual_decode_calls\":" << run.calls << ",\"actual_input_rows\":" << run.input_rows
                      << ",\"reader_numerical_qualification\":\"NOT_QUALIFIED\"}\n";
            return 0;
        }
        const auto red = token(" red"), blue = token(" blue");
        const auto green = token(" green"), orange = token(" orange");
        require(red != blue && green != orange, "fixture peer alternatives are not distinct token IDs");
        const auto run = [&](llama_token a, llama_token b, behavior mode) {
            return run_case(model.get(), threads, cache_type, base, own, a, b, mode);
        };
        behavior hidden; hidden.visible = false;
        auto ordinary = hidden; ordinary.ordinary = true;
        auto reversed = behavior{}; reversed.reverse = true;
        auto serial = behavior{}; serial.serial = true;
        auto hidden_serial = hidden; hidden_serial.serial = true;
        auto ordinary_serial = ordinary; ordinary_serial.serial = true;
        auto reversed_serial = serial; reversed_serial.reverse = true;
        auto invalid = behavior{}; invalid.invalid = true;
        auto retire = behavior{}; retire.retire = true;
        auto restore = behavior{}; restore.restore = true;
        const auto hidden_a = run(red, green, hidden);
        const auto hidden_b = run(blue, orange, hidden);
        const auto stock = run(red, green, ordinary);
        const auto visible_a = run(red, green, {});
        const auto visible_b = run(blue, green, {});
        const auto pending_b = run(red, orange, {});
        const auto stock_serial = run(red, green, ordinary_serial);
        const auto hidden_serial_a = run(red, green, hidden_serial);
        const auto visible_serial_a = run(red, green, serial);
        // A native quantized model can change its arithmetic path with batch
        // shape. Report that control separately, before any strict gate can
        // stop this run. These measurements do not waive an assertion or
        // qualify cross-shape numerical invariance.
        const auto diagnostic = [&](const result & reference, const result & actual,
                                    const char * name) {
            std::cerr << std::setprecision(17) << "{\"diagnostic\":\"" << name
                      << "\",\"kv_type\":\"" << ggml_type_name(cache_type)
                      << "\",\"round_max_abs\":[";
            for (std::size_t i = 0; i < 3; ++i) {
                if (i != 0) { std::cerr << ','; }
                std::cerr << max_abs(reference.reader[i], actual.reader[i]);
            }
            std::cerr << "]}\n";
        };
        diagnostic(stock, stock_serial, "ordinary_batch_vs_serial");
        diagnostic(hidden_a, hidden_serial_a, "hidden_ddvr_batch_vs_serial");
        diagnostic(stock_serial, hidden_serial_a, "hidden_ddvr_serial_vs_ordinary_serial");
        diagnostic(visible_a, visible_serial_a, "visible_ddvr_batch_vs_serial");
        double equivalence = 0;
        double cross_shape_difference = 0;
        std::size_t comparisons = 0, cross_shape_comparisons = 0;
        std::size_t calls = 0, rows = 0, rejected = 0, capacity_rejected = 0;
        const auto check = [&](const result & reference, const result & actual,
                               const std::string & name, bool cross_shape = false) {
            for (std::size_t i = 0; i < 3; ++i) {
                const auto difference = numeric.same(reference.reader[i], actual.reader[i],
                                                     name + " round " + std::to_string(i), cross_shape);
                if (cross_shape) {
                    cross_shape_difference = std::max(cross_shape_difference, difference);
                    ++cross_shape_comparisons;
                } else {
                    equivalence = std::max(equivalence, difference);
                    ++comparisons;
                }
            }
        };
        check(hidden_a, hidden_b, "hidden peer content isolation");
        check(stock, hidden_a, "ordinary attention reference");
        check(stock_serial, hidden_serial_a, "ordinary serial attention reference");
        equivalence = std::max(equivalence, numeric.same(visible_a.reader[0], visible_b.reader[0], "uncommitted peer isolation round 0"));
        equivalence = std::max(equivalence, numeric.same(visible_a.reader[1], pending_b.reader[1], "current peer isolation round 1"));
        comparisons += 2;
        const auto first_effect = max_abs(visible_a.reader[1], visible_b.reader[1]);
        const auto next_effect = max_abs(visible_a.reader[2], pending_b.reader[2]);
        numeric.effect(first_effect, "first committed peer effect round 1");
        numeric.effect(next_effect, "next committed peer effect round 2");
        for (const auto & value : {hidden_a, hidden_b, stock, visible_a, visible_b, pending_b,
                                  stock_serial, hidden_serial_a, visible_serial_a}) {
            calls += value.calls; rows += value.rows;
        }
        std::vector<std::pair<std::string, behavior>> cases = {
                {"batch row permutation", reversed},
                {"reversed serial allocation", reversed_serial}, {"invalid setter preservation", invalid},
                {"native recurrent retirement and replacement", retire}, {"restored reader continuation", restore}};
        const bool recurrent = llama_model_is_hybrid(model.get()) || llama_model_is_recurrent(model.get());
        if (recurrent) {
            auto full_pool = retire;
            full_pool.full_pool = true;
            cases.emplace_back("full native pool rejection and subsequent safe reuse", full_pool);
        }
        for (const auto & item : cases) {
            const auto value = run(red, green, item.second);
            // Reversing serial allocation changes physical key placement but
            // preserves one-row native calls. Keep it a strict same-shape
            // gate; the previous batch reference remains an explicit record.
            check(item.second.serial ? visible_serial_a : visible_a, value, item.first);
            if (item.second.serial) {
                check(visible_a, value, "legacy batch versus reversed serial allocation", true);
            }
            if (!value.restored.empty()) {
                equivalence = std::max(equivalence, numeric.same(visible_a.reader[2], value.restored, "restored continuation round 2"));
                equivalence = std::max(equivalence, numeric.same(value.reader[2], value.restored,
                                                               "restored continuation within same execution round 2"));
                comparisons += 2;
            }
            calls += value.calls; rows += value.rows; rejected += value.rejected;
            capacity_rejected += value.capacity_rejected;
        }
        // All former cross-shape claims retain their original numerical
        // threshold and observed failures. Native ordinary attention is also
        // measured, rather than assuming a quantized backend has a general
        // batch-invariance property. The optional flag makes every one of
        // these comparisons a required gate again.
        check(stock, stock_serial, "ordinary batch versus serial", true);
        check(hidden_a, hidden_serial_a, "hidden DDVR batch versus serial", true);
        check(visible_a, visible_serial_a, "serial frozen cohort", true);
        require(!recurrent || capacity_rejected == 1, "the full physical recurrent pool was not exercised");
        require(comparisons > 0 && rejected >= 6 && calls > 0, "empty or incomplete real execution gate");
        std::cout << std::setprecision(17)
                  << "{\"status\":\"" << (numeric.failures() == 0 ? "PASS" : "FAIL")
                  << "\",\"scope\":\"real full-model teacher-forced logits\""
                  << ",\"keep_going_diagnostics\":" << (keep_going ? "true" : "false")
                  << ",\"failed_comparisons\":" << numeric.failures()
                  << ",\"hybrid_model\":" << (llama_model_is_hybrid(model.get()) ? "true" : "false")
                  << ",\"kv_type\":\"" << ggml_type_name(cache_type) << '"'
                  << ",\"batch_invariant_matmul\":true"
                  << ",\"logical_sequences_requested\":8,\"physical_recurrent_slots_requested\":3"
                  << ",\"q_heads\":" << llama_model_n_head(model.get())
                  << ",\"kv_heads\":" << llama_model_n_head_kv(model.get())
                  << ",\"logit_comparisons\":" << comparisons
                  << ",\"equivalence_max_abs\":" << equivalence
                  << ",\"cross_shape_logit_comparisons\":" << cross_shape_comparisons
                  << ",\"cross_shape_max_abs\":" << cross_shape_difference
                  << ",\"first_peer_effect_max_abs\":" << first_effect
                  << ",\"next_peer_effect_max_abs\":" << next_effect
                  << ",\"invalid_views_rejected\":" << rejected
                  << ",\"full_recurrent_pool_rejections\":" << capacity_rejected
                  << ",\"actual_decode_calls\":" << calls << ",\"actual_input_rows\":" << rows
                  << ",\"numerical_comparisons\":";
        numeric.print(std::cout);
        std::cout << ",\"qualifications\":";
        numeric.print_qualifications(std::cout);
        std::cout << "}\n";
        return numeric.failures() == 0 ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "RERoT reader-view test failed: " << error.what() << '\n';
        return 1;
    }
}
