#include "capacity.h"
#include "native.h"
#include "options.h"
#include "sampling.h"
#include "text.h"
#include "usage.h"

#include <ggml-backend.h>
#include <tq/rerot/episode.h>
#include <tq/text/stream.h>

#include <algorithm>
#include <csignal>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
using namespace tq::rerot_compose;
using tq::rerot::StageId;
using tq::rerot::RunId;
using tq::rerot::SegmentKind;
using tq::rerot::Visibility;

volatile std::sig_atomic_t stop_requested = 0;
extern "C" void request_stop(int) { stop_requested = 1; }

class backend_owner {
public:
    backend_owner() { ggml_backend_load_all(); llama_backend_init(); }
    ~backend_owner() { llama_backend_free(); }
    backend_owner(const backend_owner &) = delete;
    backend_owner & operator=(const backend_owner &) = delete;
};

tq::rerot::Plan make_plan(const options & config) {
    std::vector<tq::rerot::Node> nodes;
    std::unordered_map<std::string, StageId> ids;
    for (const auto & item : config.tasks) {
        if (nodes.size() >= std::size_t(UINT32_MAX - 2)) { throw std::invalid_argument("too many stages"); }
        ids.emplace(item.id, StageId(nodes.size() + 1));
        nodes.push_back({item.id, item.instruction});
    }
    std::vector<tq::rerot::Edge> edges;
    for (const auto & edge : config.edges) { edges.push_back({ids.at(edge.first), ids.at(edge.second)}); }
    return tq::rerot::Plan(std::move(nodes), std::move(edges));
}

RunId run_id(StageId stage, SegmentKind kind) {
    const std::uint64_t offset = kind == SegmentKind::frame ? 0 : kind == SegmentKind::body ? 1 : 2;
    return UINT64_C(2) + std::uint64_t(stage) * 3 + offset;
}

struct stage_session {
    std::vector<llama_token> frame;
    std::unique_ptr<stage_sampler> sampler;
    std::unique_ptr<tq::text::generation_stream> stream;
    std::optional<llama_token> pending;
    std::uint64_t processed = 0;
    std::uint64_t body_tokens = 0;
    std::string text;
};

class composition {
public:
    explicit composition(options config)
        : config_(std::move(config)), plan_(make_plan(config_)),
          pens_(std::uint32_t(std::min(plan_.worker_count(), std::size_t(config_.parallel)))),
          native_(config_, std::uint32_t(plan_.worker_count() + 2), pens_,
                  []() noexcept { return stop_requested != 0; }),
          framing_(native_.model(), config_.chat_template, config_.prompt),
          sessions_(plan_.worker_count() + 2) {
        const auto & base = framing_.base();
        episode_capacity capacity(base.size());
        for (StageId stage = 1; stage <= plan_.synthesis_stage(); ++stage) {
            auto & session = sessions_.at(stage);
            session.frame = framing_.frame(stage == plan_.synthesis_stage() ?
                config_.synthesis : plan_.node(stage).intent);
            capacity.add_stage(session.frame.size(),
                stage == plan_.synthesis_stage() ? config_.final_tokens : config_.worker_tokens);
        }
        capacity.require(native_.kv_capacity(), native_.context_size());
        tq::rerot::Config episode_config;
        episode_config.episode = 1;
        episode_config.pens = pens_;
        episode_config.max_tokens = native_.kv_capacity();
        episode_ = std::make_unique<tq::rerot::Episode>(plan_, episode_config,
            std::vector<tq::rerot::BaseRun>{{1, 0, std::uint64_t(base.size())}});
        if (config_.json) {
            std::cout << "{\"event\":\"config\",\"provider\":\"native-a\",\"submission_domains\":1,\"profile\":\"assistant-eot\""
                      << ",\"workers\":" << plan_.worker_count() << ",\"parallel\":" << pens_
                      << ",\"base_tokens\":" << base.size()
                      << ",\"logical_context\":" << native_.context_size()
                      << ",\"physical_kv\":" << native_.kv_capacity()
                      << ",\"required_physical_kv\":" << capacity.physical_tokens()
                      << ",\"required_reader_context\":" << capacity.reader_tokens()
                      << ",\"logical_sequences\":" << native_.sequence_capacity()
                      << ",\"physical_recurrent_slots_requested\":" << pens_ + 1
                      << ",\"split_mode_requested\":" << json_string(config_.split)
                      << ",\"devices_requested\":" << json_string(config_.devices)
                      << ",\"devices\":" << native_.device_summary()
                      << ",\"native_context\":" << native_.context_summary()
                      << ",\"gpu_layers_requested\":" << config_.gpu_layers
                      << ",\"phase_memory\":" << (config_.phase_memory ? "true" : "false")
                      << ",\"seed\":" << config_.seed << ",\"temperature\":" << config_.temperature
                      << ",\"synthesis_stage\":" << plan_.synthesis_stage() << "}\n";
            flush();
        }
        native_.prefill(base);
    }

    void run() {
        while (episode_->state() != tq::rerot::EpisodeState::finished) {
            if (stop_requested) { throw std::runtime_error("cancelled before the next frontier"); }
            auto step = episode_->begin_step();
            try { advance(step); }
            catch (...) {
                // native_execution retains input owners through real A drain.
                // It never retries a partially executed call. The exception
                // terminates this context and no new view can be published.
                if (episode_->state() == tq::rerot::EpisodeState::computing) {
                    episode_->abort(step);
                    if (native_.drained()) { episode_->confirm_drained(step); }
                }
                throw;
            }
        }
        if (stop_requested) { throw std::runtime_error("cancelled before final output"); }
        const auto & final = sessions_.at(plan_.synthesis_stage());
        if (config_.json) {
            std::cout << "{\"event\":\"result\",\"complete\":true,\"revision\":" << episode_->revision()
                      << ",\"pipeline_parallel_enabled\":" << (native_.pipeline_parallel_enabled() ? "true" : "false")
                      << ",\"body_tokens\":" << final.body_tokens
                      << ",\"text\":" << json_string(final.text) << "}\n";
        } else { std::cout << '\n'; }
        flush();
    }

private:
    void flush() {
        std::cout.flush();
        if (!std::cout) { throw std::runtime_error("output write failed after the last committed frontier"); }
    }

    void display(StageId stage, const tq::text::stream_delta & delta) {
        auto & session = sessions_.at(stage);
        session.text += delta.text;
        if (config_.json) {
            if (!delta.text.empty() || delta.finished) {
                std::cout << "{\"event\":\"text\",\"stage\":" << stage
                          << ",\"offset\":" << delta.offset << ",\"text\":" << json_string(delta.text)
                          << ",\"finished\":" << (delta.finished ? "true" : "false") << "}\n";
            }
        } else if (stage == plan_.synthesis_stage()) { std::cout << delta.text; }
    }

    void advance(const tq::rerot::Step & step) {
        std::vector<input_row> rows;
        std::unordered_map<StageId, std::size_t> row_counts;
        bool has_frame = false;
        for (const auto & member : step.members()) {
            auto & session = sessions_.at(member.stage);
            const bool starting = member.state == tq::rerot::StageState::starting;
            SegmentKind kind = SegmentKind::frame;
            std::vector<llama_token> input;
            if (starting) {
                has_frame = true;
                native_.start(member.stage);
                session.sampler = std::make_unique<stage_sampler>(config_.seed, member.stage,
                                                                 config_.top_k, config_.temperature);
                session.stream = std::make_unique<tq::text::generation_stream>();
                for (const auto token : framing_.base()) { session.sampler->accept(token); }
                input = session.frame;
            } else {
                if (!session.pending) { throw std::logic_error("running stage has no pending token"); }
                const bool end = llama_vocab_is_eog(native_.vocab(), *session.pending);
                kind = end ? SegmentKind::source_end : SegmentKind::body;
                const auto limit = member.stage == plan_.synthesis_stage() ? config_.final_tokens : config_.worker_tokens;
                if (!end && session.body_tokens >= limit) {
                    throw std::runtime_error("stage " + std::to_string(member.stage) +
                                             " reached its token limit without source end");
                }
                input.push_back(*session.pending);
            }
            const auto position = std::uint64_t(framing_.base().size()) + session.processed;
            if (position + input.size() > native_.context_size() || position > INT32_MAX) {
                throw std::runtime_error("stage position exceeds the native context");
            }
            const auto prepared = episode_->prepare(step, member.stage,
                {{run_id(member.stage, kind), kind,
                  kind == SegmentKind::source_end ? Visibility::private_kv : Visibility::public_kv,
                  std::int64_t(position), std::uint64_t(input.size())}});
            if (prepared.size() != 1 || prepared[0].count != input.size()) {
                throw std::logic_error("B returned a different prepared row count");
            }
            row_counts.emplace(member.stage, input.size());
            for (std::size_t index = 0; index < input.size(); ++index) {
                const auto query = tq::rerot::QueryPoint{prepared[0].run, prepared[0].offset + index};
                rows.push_back({member.stage, llama_pos(position + index), input[index],
                    index + 1 == input.size() && kind != SegmentKind::source_end,
                    episode_->view(step, member.stage, query)});
            }
        }
        if (config_.phase_memory && !has_frame && previous_frame_) {
            const auto before = native_.compute_bytes();
            native_.release_compute();
            if (config_.json) {
                std::cout << "{\"event\":\"compute_release\",\"before\":" << before
                          << ",\"after\":" << native_.compute_bytes() << "}\n";
            }
        }
        previous_frame_ = has_frame;
        auto outputs = native_.forward(step.frontier(), rows);
        if (stop_requested) { throw std::runtime_error("cancelled after native completion and before publication"); }
        for (const auto & member : step.members()) {
            if (member.state == tq::rerot::StageState::starting) {
                auto & session = sessions_.at(member.stage);
                for (const auto token : session.frame) { session.sampler->accept(token); }
            }
        }
        std::unordered_map<StageId, llama_token> next;
        for (const auto & output : outputs) {
            if (!next.emplace(output.stage, sessions_.at(output.stage).sampler->sample(output.logits)).second) {
                throw std::logic_error("duplicate stage output");
            }
        }
        for (const auto & member : step.members()) {
            auto & session = sessions_.at(member.stage);
            const bool end = member.state != tq::rerot::StageState::starting && session.pending &&
                             llama_vocab_is_eog(native_.vocab(), *session.pending);
            if (!end && !next.count(member.stage)) { throw std::logic_error("missing stage logits"); }
            const auto origin = !end ? tq::rerot::EndOrigin::none :
                member.stage == plan_.synthesis_stage() ? tq::rerot::EndOrigin::synthesis_source :
                                                         tq::rerot::EndOrigin::worker_source;
            episode_->complete(step, member.stage, {row_counts.at(member.stage), origin});
        }
        // Sampling can take time after native completion. Cancellation during
        // that work still aborts the complete, unpublished cohort.
        if (stop_requested) { throw std::runtime_error("cancelled before frontier publication"); }
        const auto committed = episode_->commit(step);
        if (config_.json) {
            std::cout << "{\"event\":\"frontier\",\"frontier\":" << step.frontier()
                      << ",\"read_revision\":" << step.read_revision() << ",\"revision\":" << committed.revision
                      << ",\"pipeline_parallel_enabled\":" << (native_.pipeline_parallel_enabled() ? "true" : "false")
                      << ",\"rows\":" << rows.size() << ",\"readers\":[";
            bool first = true;
            for (const auto & member : step.members()) {
                const auto last = std::find_if(rows.rbegin(), rows.rend(), [&](const input_row & row) {
                    return row.stage == member.stage;
                });
                if (!first) { std::cout << ','; } first = false;
                std::cout << "{\"stage\":" << member.stage << ",\"pen\":" << member.pen
                          << ",\"starting\":" << (member.state == tq::rerot::StageState::starting ? "true" : "false")
                          << ",\"keys\":" << last->view.key_count()
                          << ",\"query_virtual\":" << last->view.query_virtual_pos() << ",\"spans\":[";
                bool first_span = true;
                for (const auto & span : last->view.spans()) {
                    if (!first_span) { std::cout << ','; } first_span = false;
                    std::cout << "{\"stage\":" << span.stage << ",\"run\":" << span.run
                              << ",\"offset\":" << span.offset << ",\"count\":" << span.count
                              << ",\"storage\":" << span.storage_pos << ",\"virtual\":" << span.virtual_pos << '}';
                }
                std::cout << "]}";
            }
            std::cout << "]}\n";
        }
        for (const auto & member : step.members()) {
            auto & session = sessions_.at(member.stage);
            session.processed += row_counts.at(member.stage);
            if (member.state != tq::rerot::StageState::starting) {
                const auto token = *session.pending;
                const bool end = llama_vocab_is_eog(native_.vocab(), token);
                if (config_.json) {
                    std::cout << "{\"event\":\"token\",\"frontier\":" << step.frontier()
                              << ",\"revision\":" << committed.revision << ",\"stage\":" << member.stage
                              << ",\"token\":" << token << ",\"source_end\":" << (end ? "true" : "false") << "}\n";
                }
                if (!end) { ++session.body_tokens; display(member.stage, session.stream->push(piece(native_.vocab(), token))); }
            }
            session.pending = next.count(member.stage) ? std::optional<llama_token>(next.at(member.stage)) : std::nullopt;
        }
        for (const auto stage : committed.sealed) {
            auto & session = sessions_.at(stage);
            native_.seal(stage);
            display(stage, session.stream->finish());
            const auto sampling = session.sampler->stats();
            session.sampler.reset();
            if (config_.json) {
                std::cout << "{\"event\":\"sealed\",\"stage\":" << stage
                          << ",\"sampling\":{\"compact_rows\":" << sampling.compact_rows
                          << ",\"full_rows\":" << sampling.full_rows << "}}\n";
            }
        }
        flush();
    }

    options config_;
    tq::rerot::Plan plan_;
    std::uint32_t pens_;
    native_execution native_;
    framing framing_;
    std::vector<stage_session> sessions_;
    std::unique_ptr<tq::rerot::Episode> episode_;
    bool previous_frame_ = true;
};
} // namespace

int main(int argc, char ** argv) {
#ifdef SIGPIPE
    // Let ostream report a broken pipe through the normal error/cleanup path.
    // This also covers --help and validation errors before model loading.
    std::signal(SIGPIPE, SIG_IGN);
#endif
    bool json = false;
    for (int index = 1; index < argc; ++index) { if (std::string(argv[index]) == "--json") { json = true; } }
    try {
        auto config = tq::rerot_compose::parse(argc, argv);
        if (config.help) {
            tq::rerot_compose::usage(std::cout);
            std::cout.flush();
            return std::cout ? 0 : 1;
        }
        // Validate the task graph before loading the model or opening a device.
        static_cast<void>(make_plan(config));
        std::signal(SIGINT, request_stop);
        std::signal(SIGTERM, request_stop);
        backend_owner backend;
        composition run(std::move(config));
        run.run();
        return 0;
    } catch (const std::exception & error) {
        if (json) {
            std::cout << "{\"event\":\"error\",\"complete\":false,\"message\":"
                      << tq::rerot_compose::json_string(error.what()) << "}\n";
            std::cout.flush();
        }
        std::cerr << "RERoT: " << error.what() << '\n';
        return stop_requested ? 130 : 1;
    }
}
