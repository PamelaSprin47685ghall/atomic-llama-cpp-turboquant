#include <cstdio>
#include <cstdint>
#include <type_traits>

#include "tq/alloc/events.h"

using namespace tq::alloc;

namespace {

int failures = 0;

#define CHECK(expr)                                                            \
    do {                                                                       \
        if (!(expr)) {                                                         \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__,        \
                         __LINE__, #expr);                                     \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

static_assert(std::is_trivially_copyable<AllocEvent>::value,
              "AllocEvent must be trivially copyable");
static_assert(std::is_trivially_copyable<PlanEvent>::value,
              "PlanEvent must be trivially copyable");
static_assert(std::is_trivially_copyable<LegManifest>::value,
              "LegManifest must be trivially copyable");

void test_alloc_event_value_semantics() {
    AllocEvent event;
    CHECK(event.ts_ns == 0);
    CHECK(event.leg_id == 0);
    CHECK(event.seq == 0);
    CHECK(event.lifetime == Lifetime::process);
    CHECK(event.reuse_class == ReuseClass::frame);
    CHECK(event.site == 0);
    CHECK(event.bytes == 0);
    CHECK(event.slot == 0);
    CHECK(event.slot_gen == 0);
    CHECK(event.arena_gen == 0);
    CHECK(event.window == 0);
    CHECK(event.request == 0);
    CHECK(event.action == AllocAction::lease);
    CHECK(event.from == State::free);
    CHECK(event.to == State::free);

    event.ts_ns = 1000;
    event.leg_id = 3;
    event.seq = 7;
    event.lifetime = Lifetime::step;
    event.reuse_class = ReuseClass::staging;
    event.site = 42;
    event.bytes = 4096;
    event.slot = 1;
    event.slot_gen = 2;
    event.arena_gen = 5;
    event.window = 4;
    event.request = 9;
    event.action = AllocAction::quarantine;
    event.from = State::leased;
    event.to = State::quarantined;

    AllocEvent copy = event;
    copy.seq = 8;
    copy.to = State::free;
    CHECK(event.seq == 7);
    CHECK(event.to == State::quarantined);
    CHECK(copy.seq == 8);
    CHECK(copy.to == State::free);
    CHECK(copy.lifetime == Lifetime::step);
}

void test_plan_event_value_semantics() {
    PlanEvent event;
    CHECK(event.ts_ns == 0);
    CHECK(event.leg_id == 0);
    CHECK(event.seq == 0);
    CHECK(event.plan_key.n_tokens == 1);
    CHECK(event.action == PlanAction::miss);
    CHECK(event.graph_exec_create == 0);
    CHECK(event.graph_exec_destroy == 0);
    CHECK(event.relay_uid == 0);
    CHECK(event.arena_gen == 0);

    event.ts_ns = 2000;
    event.leg_id = 4;
    event.seq = 11;
    event.plan_key.model = 12;
    event.plan_key.window = 2;
    event.plan_key.n_tokens = 4;
    event.plan_key.n_outputs = 8;
    event.plan_key.rs_class = -2;
    event.plan_key.n_kv_bucket = 255;
    event.plan_key.alloc_gen = 19;
    event.action = PlanAction::replace;
    event.graph_exec_create = 21;
    event.graph_exec_destroy = 22;
    event.relay_uid = 23;
    event.arena_gen = 24;

    PlanEvent copy = event;
    copy.action = PlanAction::hit;
    copy.plan_key.model = 13;
    CHECK(event.action == PlanAction::replace);
    CHECK(event.plan_key.model == 12);
    CHECK(copy.plan_key.model == 13);
    CHECK(copy.plan_key.n_kv_bucket == 255);
}

void test_leg_manifest_value_semantics() {
    LegManifest manifest;
    CHECK(manifest.leg_id == 0);
    CHECK(manifest.build_sha == 0);
    CHECK(manifest.closure4[0] == 0);
    CHECK(manifest.closure4[1] == 0);
    CHECK(manifest.closure4[2] == 0);
    CHECK(manifest.closure4[3] == 0);
    CHECK(manifest.model_sha == 0);
    CHECK(manifest.env_sha == 0);
    CHECK(manifest.platform_fingerprint == 0);
    CHECK(manifest.loadavg == 0.0);
    CHECK(!manifest.gpu_idle);
    CHECK(manifest.iova_last == 0);
    CHECK(manifest.decode_submit_p50 == 0.0);
    CHECK(manifest.decode_submit_p95 == 0.0);
    CHECK(manifest.tok_s == 0.0);
    CHECK(!manifest.all_511);
    CHECK(manifest.ttft == 0.0);
    CHECK(manifest.generation_span == 0);
    CHECK(manifest.verdict == Verdict::unknown);

    manifest.leg_id = 7;
    manifest.build_sha = 0x1234;
    manifest.closure4[0] = 1;
    manifest.closure4[1] = 2;
    manifest.closure4[2] = 3;
    manifest.closure4[3] = 4;
    manifest.model_sha = 0x5678;
    manifest.env_sha = 0x9abc;
    manifest.platform_fingerprint = 0xdef0;
    manifest.loadavg = 1.5;
    manifest.gpu_idle = true;
    manifest.iova_last = 4096;
    manifest.decode_submit_p50 = 0.25;
    manifest.decode_submit_p95 = 0.75;
    manifest.tok_s = 31.5;
    manifest.all_511 = true;
    manifest.ttft = 0.125;
    manifest.generation_span = 9;
    manifest.verdict = Verdict::pass;

    LegManifest copy = manifest;
    copy.closure4[0] = 99;
    copy.verdict = Verdict::fail;
    CHECK(manifest.closure4[0] == 1);
    CHECK(manifest.verdict == Verdict::pass);
    CHECK(copy.closure4[0] == 99);
    CHECK(copy.verdict == Verdict::fail);
    CHECK(copy.tok_s == 31.5);
    CHECK(copy.gpu_idle);
}

} // namespace

int main() {
    test_alloc_event_value_semantics();
    test_plan_event_value_semantics();
    test_leg_manifest_value_semantics();

    if (failures != 0) {
        std::fprintf(stderr, "%d event checks failed\n", failures);
        return 1;
    }
    std::printf("event tests passed\n");
    return 0;
}
