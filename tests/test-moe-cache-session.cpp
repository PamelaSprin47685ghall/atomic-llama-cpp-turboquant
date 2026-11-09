// Scheduler MoE-cache session policy: a scheduler created by
// ggml_backend_sched_new owns an environment-default session. Re-applying the
// context's resolved policy must replace it; OFF must remove it; a dormant
// request must leave no session; and an unproven scheduler must refuse the
// change without destroying or replacing the current session.
//
// This test uses a real ggml_backend_sched over the CPU backend and a real
// provider implementation with real session objects. It observes the provider's
// session creation/destruction lifecycle, not a scheduler getter: the default
// session must be destroyed before a configured one is created, and a dormant
// or refused request must not leave the environment default behind.

#include "ggml.h"
#include "ggml-backend.h"

#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-backend-moe-cache.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

struct session_record {
    bool   is_default   = false;
    int    automatic    = -1;
    size_t budget_bytes = 0;
};

std::vector<session_record> g_created;
std::vector<session_record> g_destroyed;
bool                        g_fail_create = false;

int query_config(int automatic, size_t budget_mib, ggml_moe_cache_config * config) {
    if (config == nullptr) {
        return 0;
    }
    std::memset(config, 0, sizeof(*config));
    config->budget_bytes         = budget_mib * 1024 * 1024;
    config->reserve_bytes        = 0;
    config->minimum_slab_bytes   = automatic > 0 ? 1 : 0;
    config->min_expert_bytes     = 1;
    config->min_expert_explicit  = 1;
    config->max_batch            = 1;
    config->min_devices          = 1;
    config->min_compute_capability = 0;
    config->overlap_cpu_rows     = -1;
    return 1;
}

void * session_create(void * const *, int, const ggml_moe_cache_config * config) {
    if (g_fail_create) {
        return nullptr;
    }
    auto * rec         = new session_record();
    rec->is_default    = config == nullptr;
    if (config != nullptr) {
        rec->budget_bytes = config->budget_bytes;
        rec->automatic    = config->minimum_slab_bytes > 0 ? 1 : 0;
    }
    g_created.push_back(*rec);
    return rec;
}

void session_destroy(void * opaque) {
    auto * rec = static_cast<session_record *>(opaque);
    if (rec != nullptr) {
        g_destroyed.push_back(*rec);
        delete rec;
    }
}

void session_enter(void *) {}
void session_leave(void *) {}

int g_checks   = 0;
int g_failures = 0;

void check(bool condition, const char * message) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
}

constexpr size_t MiB = 1024 * 1024;

} // namespace

int main() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu == nullptr) {
        std::fprintf(stderr, "SKIP: no CPU backend\n");
        return 77;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(cpu);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);

    ggml_moe_cache_api api = {};
    api.owner            = reg;
    api.query_config     = query_config;
    api.session_create   = session_create;
    api.session_destroy  = session_destroy;
    api.session_enter    = session_enter;
    api.session_leave    = session_leave;
    ggml_moe_cache_register(&api);

    ggml_backend_t backends[] = { cpu };
    auto make_sched = [&]() {
        return ggml_backend_sched_new(backends, nullptr, 1, 1024, false, false);
    };

    // A scheduler is born with the environment-default session.
    ggml_backend_sched_t a = make_sched();
    check(a != nullptr, "scheduler A created");
    check(g_created.size() == 1 && g_created.back().is_default, "A creates the default session");

    // OFF must destroy the default session and create no replacement.
    check(ggml_backend_sched_set_moe_cache(a, GGML_MOE_CACHE_MODE_OFF, 0) == GGML_MOE_CACHE_SET_APPLIED,
          "A off applied");
    check(g_destroyed.size() == 1 && g_destroyed.back().is_default, "A off destroys the default session");
    check(g_created.size() == 1, "A off creates no session");

    // AUTO replaces the (already gone) default with a configured session.
    check(ggml_backend_sched_set_moe_cache(a, GGML_MOE_CACHE_MODE_AUTO, 4096) == GGML_MOE_CACHE_SET_APPLIED,
          "A auto applied");
    check(g_destroyed.size() == 1, "A auto destroys no stale session");
    check(g_created.size() == 2 && !g_created.back().is_default, "A auto creates a configured session");
    check(g_created.back().budget_bytes == 4096 * MiB, "A auto config budget");
    check(g_created.back().automatic == 1, "A auto config automatic");

    // An explicit ON replaces the AUTO session with a non-automatic session.
    check(ggml_backend_sched_set_moe_cache(a, GGML_MOE_CACHE_MODE_ON, 1234) == GGML_MOE_CACHE_SET_APPLIED,
          "A on applied");
    check(g_destroyed.size() == 2 && !g_destroyed.back().is_default, "A on destroys the AUTO session");
    check(g_created.size() == 3 && !g_created.back().is_default, "A on creates a configured session");
    check(g_created.back().budget_bytes == 1234 * MiB, "A on config budget");
    check(g_created.back().automatic == 0, "A on config not automatic");

    // B simulates a scheduler recreated by a reserve retry or a KV-stream
    // phase switch. The new default session must be destroyed and replaced by
    // the same configured policy.
    ggml_backend_sched_t b = make_sched();
    check(b != nullptr, "scheduler B created");
    check(g_created.size() == 4 && g_created.back().is_default, "B creates the default session");
    check(ggml_backend_sched_set_moe_cache(b, GGML_MOE_CACHE_MODE_AUTO, 4096) == GGML_MOE_CACHE_SET_APPLIED,
          "B auto applied");
    check(g_destroyed.size() == 3 && g_destroyed.back().is_default, "B auto destroys the default session");
    check(g_created.size() == 5 && !g_created.back().is_default, "B auto creates a configured session");
    check(g_created.back().budget_bytes == 4096 * MiB, "B auto config budget");
    check(g_created.back().automatic == 1, "B auto config automatic");

    // OFF is a user policy, not a request to fall back to the default.
    check(ggml_backend_sched_set_moe_cache(b, GGML_MOE_CACHE_MODE_OFF, 0) == GGML_MOE_CACHE_SET_APPLIED,
          "B off applied");
    check(g_destroyed.size() == 4 && !g_destroyed.back().is_default, "B off destroys the configured session");
    check(g_created.size() == 5, "B off creates no session");

    // Dormant: the default session must be destroyed and no session may
    // remain. A dormant request is not the environment default. The
    // llama_context helper treats a dormant explicit ON as a failure; AUTO
    // may stay dormant by design. This test pins the setter's DORMANT result.
    ggml_backend_sched_t d = make_sched();
    check(d != nullptr, "scheduler D created");
    check(g_created.size() == 6 && g_created.back().is_default, "D creates the default session");
    g_fail_create = true;
    check(ggml_backend_sched_set_moe_cache(d, GGML_MOE_CACHE_MODE_ON, 1) == GGML_MOE_CACHE_SET_DORMANT,
          "D dormant outcome");
    check(g_destroyed.size() == 5 && g_destroyed.back().is_default, "D dormant destroys the default session");
    check(g_created.size() == 6, "D dormant creates no session");
    g_fail_create = false;

    // Refused: an unproven scheduler keeps its current session and policy.
    ggml_backend_sched_t e = make_sched();
    check(e != nullptr, "scheduler E created");
    check(g_created.size() == 7 && g_created.back().is_default, "E creates the default session");
    check(ggml_backend_sched_set_moe_cache(e, GGML_MOE_CACHE_MODE_AUTO, 4096) == GGML_MOE_CACHE_SET_APPLIED,
          "E auto applied");
    check(g_destroyed.size() == 6 && g_destroyed.back().is_default, "E auto destroys the default session");
    check(g_created.size() == 8 && !g_created.back().is_default, "E auto creates a configured session");

    const size_t created_before   = g_created.size();
    const size_t destroyed_before = g_destroyed.size();
    ggml_backend_sched_mark_unproven(e);
    check(ggml_backend_sched_set_moe_cache(e, GGML_MOE_CACHE_MODE_OFF, 0) == GGML_MOE_CACHE_SET_REFUSED,
          "E unproven off refused");
    check(g_created.size() == created_before, "E refusal created no session");
    check(g_destroyed.size() == destroyed_before, "E refusal destroyed no session");

    // E is intentionally not freed: an unproven scheduler is process-owned and
    // ggml_backend_sched_free retains it. The process exit owns the rest.
    ggml_backend_sched_free(a);
    ggml_backend_sched_free(b);
    ggml_backend_sched_free(d);
    ggml_moe_cache_unregister(reg);
    ggml_backend_free(cpu);

    if (g_failures != 0) {
        std::fprintf(stderr, "MOE_SESSION_TEST FAIL checks=%d failures=%d\n", g_checks, g_failures);
        return 1;
    }
    std::printf("MOE_SESSION_TEST PASS checks=%d\n", g_checks);
    return 0;
}
