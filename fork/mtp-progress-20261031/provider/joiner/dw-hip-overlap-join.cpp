// DUALWIN-HIP-JOIN — offline joiner tool for device-stream overlap evidence.
//
// Reads the combined evidence log (relay provider TP5_DW_HIP_SPAN /
// TP5_DW_HIP_SPANS / TP5_DW_HIP_OVERLAP lines interleaved with the runtime's
// TP5_DW_HIP_OVERLAP_MAP lines; unrelated lines are skipped) from the given
// files or stdin, joins windows to device spans, and prints the
// TP5_DW_HIP_OVERLAP_JOINED* verdict to stdout.
//
// Exit codes: 0 verdict emitted; 1 evidence/refusal failure (reason on
// stderr, nothing on stdout); 2 usage error. Fail-closed: incomplete device
// evidence never produces a number.
//
// CPU-only tool: pure C++17, no HIP, no ggml. Built and exercised on this
// box; the device numbers it consumes are produced by relay.hip on the
// HIP machine (syntax-unverified there — no HIP toolchain on this host).

#include "dw-hip-join.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

bool feed_file(const char * path, bool stdin_path, dw_hip_join::model & m, std::string & err) {
    std::FILE * f = stdin_path ? stdin : std::fopen(path, "r");
    if (f == nullptr) {
        err = std::string("cannot open ") + path;
        return false;
    }
    std::unique_ptr<std::FILE, int (*)(std::FILE *)> close_guard(f, stdin_path ? [](std::FILE *) { return 0; } : std::fclose);
    char buf[4096];
    unsigned long long line_no = 0;
    while (std::fgets(buf, sizeof(buf), f) != nullptr) {
        ++line_no;
        const std::size_t len = std::strlen(buf);
        if (len == sizeof(buf) - 1 && buf[len - 1] != '\n') {
            err = "line " + dw_hip_join::detail::u64(line_no) + " exceeds the joiner line buffer (evidence line truncated)";
            return false;
        }
        try {
            (void) m.feed_line(buf, line_no);
        } catch (const dw_hip_join::parse_error & e) {
            err = std::string("parse error at line ") + dw_hip_join::detail::u64(e.line) + ": " + e.what();
            return false;
        }
    }
    if (std::ferror(f) != 0) {
        err = std::string("read error on ") + (stdin_path ? "<stdin>" : path);
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    std::vector<const char *> paths;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            std::fprintf(stderr, "usage: dw-hip-overlap-join [LOGFILE|- ...]\n"
                                 "reads the combined TP5_DW_HIP_* evidence log (default stdin)\n");
            return 2;
        }
        paths.push_back(argv[i]);
    }
    dw_hip_join::model m;
    if (paths.empty()) {
        std::string err;
        if (!feed_file("<stdin>", true, m, err)) {
            std::fprintf(stderr, "TP5_DW_HIP_OVERLAP_JOINED_ERROR reason=%s\n", err.c_str());
            return 1;
        }
    } else {
        for (const char * path : paths) {
            std::string err;
            const bool stdin_path = std::strcmp(path, "-") == 0;
            if (!feed_file(path, stdin_path, m, err)) {
                std::fprintf(stderr, "TP5_DW_HIP_OVERLAP_JOINED_ERROR reason=%s\n", err.c_str());
                return 1;
            }
        }
    }
    try {
        m.finalize();
        const dw_hip_join::verdict v = dw_hip_join::analyze(m);
        (void) dw_hip_join::print_verdict(v, stdout);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "TP5_DW_HIP_OVERLAP_JOINED_ERROR reason=%s\n", e.what());
        return 1;
    }
    return 0;
}
