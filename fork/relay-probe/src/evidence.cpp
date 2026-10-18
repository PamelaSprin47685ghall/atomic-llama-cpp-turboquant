#include "evidence.h"

#include <cstdio>

namespace tq::relay::probe {

namespace {

const char * outcome_names[] = { "pass", "fail", "unsupported", "not_run" };

const char * case_names[] = {
    "device_identity",         "mapped_host_access",      "cpu_to_gpu_publish",     "gpu_to_cpu_publish",
    "producer_delay_ordering", "ticket_semantics",        "multi_group_completion", "abort_stop",
    "host_system_rmw_u64_add", "host_system_rmw_u64_cas", "five_rank_cross_device",
};

const char * name_of(const char * const * names, std::size_t count, std::size_t index) noexcept {
    return index < count ? names[index] : "unknown";
}

void write_string(std::string & out, const std::string & value) {
    out += '"';
    for (const unsigned char byte : value) {
        switch (byte) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (byte < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", byte);
                    out += buffer;
                } else {
                    out += static_cast<char>(byte);
                }
        }
    }
    out += '"';
}

void write_key(std::string & out, const char * key) {
    write_string(out, key);
}

}  // namespace

const char * to_string(outcome value) noexcept {
    return name_of(outcome_names, sizeof(outcome_names) / sizeof(outcome_names[0]), static_cast<std::size_t>(value));
}

const char * case_name(case_id value) noexcept {
    return name_of(case_names, sizeof(case_names) / sizeof(case_names[0]), static_cast<std::size_t>(value));
}

std::size_t case_count() noexcept {
    return sizeof(case_names) / sizeof(case_names[0]);
}

std::string to_json(const probe_report & report) {
    std::string out;
    out += '{';
    bool first     = true;
    auto separator = [&]() {
        if (!first) {
            out += ',';
        }
        first = false;
    };
    auto integer = [&](const char * key, std::uint64_t value) {
        separator();
        write_key(out, key);
        out += ':' + std::to_string(value);
    };
    auto text = [&](const char * key, const std::string & value) {
        separator();
        write_key(out, key);
        out += ':';
        write_string(out, value);
    };
    integer("schema", report.schema);
    text("target", report.target);
    text("build_id", report.build_id);
    integer("hip_runtime_version", report.hip_runtime_version);
    text("hip_runtime_version_string", report.hip_runtime_version_string);
    text("compiler_version", report.compiler_version);
    text("host_arch", report.host_arch);
    integer("control_word_bits", report.control_word_bits);
    text("memory_flags_requested", report.memory_flags_requested);
    integer("hot_reuse_rounds", report.hot_reuse_rounds);
    separator();
    write_key(out, "devices");
    out += ":[";
    for (std::size_t i = 0; i < report.devices.size(); ++i) {
        if (i != 0) {
            out += ',';
        }
        const auto & device = report.devices[i];
        out += '{';
        bool device_first     = true;
        auto device_separator = [&]() {
            if (!device_first) {
                out += ',';
            }
            device_first = false;
        };
        auto device_integer = [&](const char * key, std::uint64_t value) {
            device_separator();
            write_key(out, key);
            out += ':' + std::to_string(value);
        };
        auto device_text = [&](const char * key, const std::string & value) {
            device_separator();
            write_key(out, key);
            out += ':';
            write_string(out, value);
        };
        device_integer("rank", static_cast<std::uint64_t>(device.ordinal));
        device_text("pci_bdf", device.pci_bus_id);
        device_text("reported_name", device.name);
        device_text("gcn_arch_name", device.gcn_arch);
        device_integer("total_vram_bytes", device.total_vram_bytes);
        device_separator();
        write_key(out, "can_map_host_memory");
        out += device.can_map_host_memory ? ":true" : ":false";
        device_separator();
        write_key(out, "cases");
        out += ":{";
        for (std::size_t j = 0; j < device.cases.size(); ++j) {
            if (j != 0) {
                out += ',';
            }
            write_key(out, case_name(device.cases[j].id));
            out += ":{\"result\":";
            write_string(out, to_string(device.cases[j].result));
            out += ",\"detail\":";
            write_string(out, device.cases[j].detail);
            out += '}';
        }
        out += "}}";
    }
    out += ']';
    text("five_rank_status", to_string(report.five_rank_status));
    text("five_rank_detail", report.five_rank_detail);
    text("qualification", to_string(report.qualification));
    text("qualification_detail", report.qualification_detail);
    separator();
    write_key(out, "limits");
    out += ":[";
    for (std::size_t i = 0; i < report.limits.size(); ++i) {
        if (i != 0) {
            out += ',';
        }
        write_string(out, report.limits[i]);
    }
    out += "]}";
    return out;
}

}  // namespace tq::relay::probe
