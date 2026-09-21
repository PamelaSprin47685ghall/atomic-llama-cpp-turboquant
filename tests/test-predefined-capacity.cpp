#include "ggml-predefined.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

// Pure CPU metadata tests. No backend enumeration, shader build, GPU context,
// model load, or hidden test execution is performed by any helper here.
#define CHECK(x) do { if (!(x)) throw std::runtime_error("failed: " #x); } while (0)

static ggml_predefined_limits limits() {
    return {GGML_PREDEFINED_ABI_VERSION, sizeof(ggml_predefined_limits), 1, 3, 32, 256,
            4, 2560, 10240, 248320, 4, 5};
}

static void capacity_and_frames() {
    char error[128];
    auto p = limits();
    ggml_predefined_capacity cap{};
    CHECK(ggml_predefined_make_capacity(&p, &cap, error, sizeof(error)));
    CHECK(cap.tokens == 32 && cap.verify_tokens == 4);
    CHECK(cap.hidden_bytes == 32ULL * 10240 * 4);
    CHECK(cap.payload_bytes == 32ULL * 2560 * 4);
    CHECK(cap.logits_bytes == 4ULL * 248320 * 4);
    CHECK(error[0] == '\0');

    const auto original = cap;
    for (int which = 0; which < 7; ++which) {
        p = limits();
        switch (which) {
            case 0: p.version++; break;
            case 1: p.sequences = 0; break;
            case 2: p.draft_tokens = UINT32_MAX; break;
            case 3: p.ubatch_tokens = 3; break;
            case 4: p.output_rows = 1; break;
            case 5: p.wire_bytes = 8; break;
            case 6: p.ranks = GGML_PREDEFINED_MAX_RANKS + 1; break;
        }
        CHECK(!ggml_predefined_make_capacity(&p, &cap, error, sizeof(error)));
        CHECK(std::memcmp(&cap, &original, sizeof(cap)) == 0);
        CHECK(error[0] != '\0');
    }
    p = limits();
    p.wire_bytes = 2;
    ggml_predefined_capacity f16{};
    CHECK(ggml_predefined_make_capacity(&p, &f16, nullptr, 0));
    CHECK(f16.payload_bytes * 2 == cap.payload_bytes);
    CHECK(f16.hidden_bytes == cap.hidden_bytes && f16.logits_bytes == cap.logits_bytes);

    ggml_predefined_request r{GGML_PREDEFINED_TARGET, 1, 4, 4, 100, 3, 0, 0};
    ggml_predefined_frame f{};
    CHECK(ggml_predefined_make_frame(&cap, &r, (1ULL << 32) + 7, 1, &f, error, sizeof(error)));
    CHECK(f.epoch_hi == 1 && f.epoch_lo == 7 && f.slot == 1);
    CHECK(f.active_tokens == 4 && f.payload_bytes == 4 * 2560 * 4);
    CHECK(f.active_tokens != cap.tokens);
    const auto good = f;
    r.tokens = 3;
    CHECK(!ggml_predefined_make_frame(&cap, &r, 2, 0, &f, error, sizeof(error)));
    CHECK(std::memcmp(&f, &good, sizeof(f)) == 0);
    r = {GGML_PREDEFINED_TARGET, 1, 1, 1, 100, 0, 0, 0};
    CHECK(ggml_predefined_make_frame(&cap, &r, 8, 0, &f, error, sizeof(error)));
    CHECK(f.payload_elements == 2560);

    for (uint32_t accepted = 0; accepted <= 3; ++accepted) {
        r = {GGML_PREDEFINED_CATCHUP, 1, accepted + 1, 0, 100, 3, accepted, 0};
        CHECK(ggml_predefined_make_frame(&cap, &r, 9 + accepted, 0, &f, error, sizeof(error)));
        CHECK(f.active_tokens == accepted + 1 && f.active_outputs == 0);
    }
    r = {GGML_PREDEFINED_DRAFT, 1, 1, 1, 100, 0, 0, 2};
    CHECK(ggml_predefined_make_frame(&cap, &r, 20, 0, &f, error, sizeof(error)));
    r.draft_step = 3;
    CHECK(!ggml_predefined_make_frame(&cap, &r, 21, 0, &f, error, sizeof(error)));
    r = {GGML_PREDEFINED_PREFILL, 1, 32, 1, 100, 0, 0, 0};
    CHECK(ggml_predefined_make_frame(&cap, &r, 22, 0, &f, error, sizeof(error)));
    CHECK(f.active_outputs == 1 && f.active_tokens == 32);
    CHECK(!ggml_predefined_make_frame(&cap, &r, 0, 0, &f, error, sizeof(error)));
    CHECK(!ggml_predefined_make_frame(&cap, &r, 23, GGML_PREDEFINED_FRAME_SLOTS, &f, error, sizeof(error)));
    r.tokens = 33;
    CHECK(!ggml_predefined_make_frame(&cap, &r, 23, 0, &f, error, sizeof(error)));
}

static void indirect_and_workspace() {
    ggml_predefined_capacity cap{};
    auto p = limits();
    CHECK(ggml_predefined_make_capacity(&p, &cap, nullptr, 0));
    ggml_predefined_request r{GGML_PREDEFINED_TARGET, 1, 4, 4, 100, 3, 0, 0};
    ggml_predefined_frame f{};
    CHECK(ggml_predefined_make_frame(&cap, &r, 1, 0, &f, nullptr, 0));

    ggml_predefined_dispatch d{};
    d.phase_mask = 1u << GGML_PREDEFINED_TARGET;
    d.axis[0] = {GGML_PREDEFINED_EXTENT_PAYLOAD, 1, 64, 65535};
    d.axis[1] = {GGML_PREDEFINED_EXTENT_ONE, 1, 1, 65535};
    d.axis[2] = d.axis[1];
    ggml_predefined_dispatch_args args{};
    CHECK(ggml_predefined_dispatch_arguments(&f, &d, &args, nullptr, 0));
    CHECK(args.x == 160 && args.y == 1 && args.z == 1);
    r = {GGML_PREDEFINED_TARGET, 1, 1, 1, 100, 0, 0, 0};
    CHECK(ggml_predefined_make_frame(&cap, &r, 2, 1, &f, nullptr, 0));
    CHECK(ggml_predefined_dispatch_arguments(&f, &d, &args, nullptr, 0));
    CHECK(args.x == 40); // NOT 1280 workgroups for all 32 capacity rows
    d.phase_mask = 1u << GGML_PREDEFINED_DRAFT;
    CHECK(ggml_predefined_dispatch_arguments(&f, &d, &args, nullptr, 0));
    CHECK(args.x == 0);
    d.axis[0].divisor = 0;
    const auto old = args;
    CHECK(!ggml_predefined_dispatch_arguments(&f, &d, &args, nullptr, 0));
    CHECK(std::memcmp(&old, &args, sizeof(args)) == 0);
    d.axis[0] = {GGML_PREDEFINED_EXTENT_PAYLOAD, UINT32_MAX, 1, 65535};
    CHECK(!ggml_predefined_dispatch_arguments(&f, &d, &args, nullptr, 0));

    ggml_predefined_workspace w{{100, 200, 80, 160}, 65, 64};
    ggml_predefined_workspace_layout layout{};
    CHECK(ggml_predefined_plan_workspace(&w, &layout, nullptr, 0));
    CHECK(layout.scratch_bytes == 256 && layout.persistent_bytes == 128);
    CHECK(layout.persistent_offset == 256 && layout.total_bytes == 384);
    const auto before = layout;
    w.alignment = 3;
    CHECK(!ggml_predefined_plan_workspace(&w, &layout, nullptr, 0));
    CHECK(std::memcmp(&before, &layout, sizeof(layout)) == 0);
    w.alignment = 64;
    w.scratch_bytes[0] = UINT64_MAX;
    CHECK(!ggml_predefined_plan_workspace(&w, &layout, nullptr, 0));
}

static void frame_slot_ownership() {
    auto slots = ggml_predefined_slots_new(5);
    CHECK(slots != nullptr);
    uint32_t first = UINT32_MAX, second = UINT32_MAX, third = 123;
    CHECK(!ggml_predefined_slots_acquire(slots, 0, &first));
    CHECK(ggml_predefined_slots_acquire(slots, 1, &first));
    CHECK(ggml_predefined_slots_record_submit(slots, first, 1, 0, 10));
    CHECK(ggml_predefined_slots_record_submit(slots, first, 1, 1, 20));
    CHECK(!ggml_predefined_slots_cancel(slots, first, 1));
    CHECK(ggml_predefined_slots_acquire(slots, 2, &second));
    CHECK(first != second);
    CHECK(!ggml_predefined_slots_record_submit(slots, second, 2, 0, 10));
    CHECK(!ggml_predefined_slots_acquire(slots, 3, &third) && third == 123);
    std::array<uint64_t, 5> observed{10, 19, 0, 0, 0};
    CHECK(!ggml_predefined_slots_retire(slots, first, 1, observed.data(), observed.size()));
    observed[1] = 20;
    CHECK(ggml_predefined_slots_retire(slots, first, 1, observed.data(), observed.size()));
    CHECK(ggml_predefined_slots_acquire(slots, 3, &third) && third == first);
    CHECK(!ggml_predefined_slots_retire(slots, third, 1, observed.data(), observed.size()));
    CHECK(!ggml_predefined_slots_retire(slots, third, 3, observed.data(), observed.size()));
    CHECK(ggml_predefined_slots_cancel(slots, third, 3));
    CHECK(ggml_predefined_slots_cancel(slots, second, 2));
    CHECK(!ggml_predefined_slots_pending(slots));
    CHECK(!ggml_predefined_slots_acquire(slots, 2, &third));
    ggml_predefined_slots_free(slots);
}

int main() {
    try {
        capacity_and_frames();
        indirect_and_workspace();
        frame_slot_ownership();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    std::puts("predefined capacity/frame/retirement checks passed");
    return 0;
}
