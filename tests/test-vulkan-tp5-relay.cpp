// Opt-in hardware PoC; deliberately not part of unattended ctest runs.
#include "ggml-vulkan.h"
#include "ggml-vulkan-relay.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <climits>
#include <dirent.h>
#include <fstream>
#include <string>
#include <vector>

static size_t count_fds() {
    DIR * d = opendir("/proc/self/fd");
    if (!d) return SIZE_MAX;
    size_t n = 0;
    while (readdir(d)) ++n;
    closedir(d);
    return n;
}

static bool idle_gpus() {
    DIR * d = opendir("/sys/class/drm");
    if (!d) return false;
    size_t checked = 0;
    bool idle = true;
    while (auto * entry = readdir(d)) {
        if (strncmp(entry->d_name, "card", 4) != 0 || entry->d_name[4] < '0' || entry->d_name[4] > '9') continue;
        const std::string path = std::string("/sys/class/drm/") + entry->d_name + "/device/";
        std::ifstream busy(path + "gpu_busy_percent");
        if (!busy) continue;
        int percent = -1;
        busy >> percent;
        unsigned long long vram = 0;
        std::ifstream(path + "mem_info_vram_used") >> vram;
        fprintf(stderr, "relay audit %s busy=%d%% vram=%llu\n", entry->d_name, percent, vram);
        // Reject unknown/active state and model residency, not just idle engines.
        idle &= percent == 0 && vram < 128ULL * 1024 * 1024;
        ++checked;
    }
    closedir(d);
    return idle && checked >= 5;
}

int main(int argc, char ** argv) {
    ggml_vk_relay_config cfg;
    bool run = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--run") == 0) { run = true; continue; }
        if (strcmp(argv[i], "--stale-doorbell") == 0) { cfg.stale_doorbell = true; continue; }
        uint32_t * field = nullptr;
        if (strcmp(argv[i], "--stages") == 0) field = &cfg.stages;
        if (strcmp(argv[i], "--replays") == 0) field = &cfg.replays;
        if (strcmp(argv[i], "--elements") == 0) field = &cfg.elements;
        if (strcmp(argv[i], "--spin-max") == 0) field = &cfg.spin_max;
        if (strcmp(argv[i], "--delay-us") == 0) field = &cfg.delay_us;
        if (strcmp(argv[i], "--withhold-stage") == 0) field = &cfg.withhold_stage;
        if (strcmp(argv[i], "--partial-submit-ranks") == 0) field = &cfg.partial_submit_ranks;
        if (!field || ++i == argc) { fprintf(stderr, "invalid/missing argument\n"); return 1; }
        char * end = nullptr;
        errno = 0;
        const unsigned long value = strtoul(argv[i], &end, 10);
        if (errno || !*argv[i] || *end || value > UINT32_MAX || argv[i][0] == '-') return 1;
        *field = (uint32_t) value;
    }
    if (!run) {
        fprintf(stderr, "Hardware experiment: test-vulkan-tp5-relay --run [--stages 1..96] [--replays 1..32]\n"
                        "  [--elements 1..4096] [--spin-max 1..1000000] [--delay-us 0..2000]\n"
                        "  [--withhold-stage N] [--stale-doorbell] [--partial-submit-ranks 1..4]\n");
        return 0;
    }
    if (!idle_gpus()) { fprintf(stderr, "relay: GPU safety audit refused run\n"); return 1; }
    const size_t before = count_fds();
    std::vector<ggml_backend_t> backends;
    ggml_backend_reg_t reg = ggml_backend_vk_reg();
    if (ggml_backend_reg_dev_count(reg) < 5) { fprintf(stderr, "relay: five GPUs required\n"); return 1; }
    for (size_t i = 0; i < 5; ++i) {
        auto * backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, i), nullptr);
        if (!backend) {
            for (auto * b : backends) ggml_backend_free(b);
            return 1;
        }
        backends.push_back(backend);
    }
    const size_t warm_fds = count_fds();
    const int result = ggml_vk_tp5_relay_probe(backends.data(), backends.size(), cfg);
    if (result == 2) {
        fprintf(stderr, "relay: FATAL undrained GPU work; no resource teardown attempted\n");
        return 2;
    }
    const size_t after_probe = count_fds();
    fprintf(stderr, "relay probe fds: before=%zu after=%zu delta=%lld\n", warm_fds, after_probe,
            (long long) after_probe - (long long) warm_fds);
    for (auto * b : backends) ggml_backend_free(b);
    const size_t after = count_fds();
    fprintf(stderr, "relay process fds: cold=%zu after=%zu (backend global caches may persist)\n", before, after);
    return result || warm_fds == SIZE_MAX || warm_fds != after_probe;
}
