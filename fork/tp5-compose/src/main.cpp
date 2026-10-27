#include <fcntl.h>
#include <sys/stat.h>
#include <tq/tp5/runtime.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void system_error(const std::string & what) {
    throw std::runtime_error(what + ": " + std::strerror(errno));
}

struct fd_owner {
    int value = -1;

    explicit fd_owner(int fd = -1) : value(fd) {}

    ~fd_owner() {
        if (value >= 0) {
            ::close(value);
        }
    }

    fd_owner(const fd_owner &)             = delete;
    fd_owner & operator=(const fd_owner &) = delete;
};

// Resolve every directory component without following a symlink. Keep this
// directory descriptor throughout the run: publication never re-resolves it.
int open_parent(const std::string & path, std::string & leaf) {
    if (path.empty() || path.find('\0') != std::string::npos || path.back() == '/') {
        throw std::invalid_argument("invalid file path");
    }
    const auto split = path.find_last_of('/');
    leaf             = split == std::string::npos ? path : path.substr(split + 1);
    if (leaf == "." || leaf == "..") {
        throw std::invalid_argument("invalid file basename");
    }
    fd_owner dir(::open(path.front() == '/' ? "/" : ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    if (dir.value < 0) {
        system_error("open parent directory");
    }
    if (split != std::string::npos) {
        std::size_t begin = path.front() == '/' ? 1 : 0;
        while (begin < split) {
            auto end = path.find('/', begin);
            if (end == std::string::npos || end > split) {
                end = split;
            }
            const auto part = path.substr(begin, end - begin);
            if (part == "..") {
                throw std::invalid_argument("parent traversal is not allowed: " + path);
            }
            if (!part.empty() && part != ".") {
                const int next = ::openat(dir.value, part.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
                if (next < 0) {
                    system_error("open parent component " + part);
                }
                ::close(dir.value);
                dir.value = next;
            }
            begin = end + 1;
        }
    }
    const int result = dir.value;
    dir.value        = -1;
    return result;
}

int open_input(const std::string & path) {
    if (path.empty() || path.find('\0') != std::string::npos) {
        throw std::invalid_argument("empty or invalid input path");
    }
    // Input is read-only. Normal model-cache symlinks and relative paths do
    // not carry the overwrite risk of publishing an output file.
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0) {
        system_error("open input " + path);
    }
    fd_owner    input(fd);
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        system_error("stat input " + path);
    }
    if (!S_ISREG(st.st_mode)) {
        throw std::invalid_argument("input must be a regular file: " + path);
    }
    input.value = -1;
    return fd;
}

std::uint64_t number(std::string_view    text,
                     const std::string & name,
                     bool                zero    = false,
                     std::uint64_t       maximum = INT32_MAX) {
    std::uint64_t result = 0;
    const auto    parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || (!zero && result == 0) ||
        result > maximum) {
        throw std::invalid_argument(name + " requires a " + (zero ? "nonnegative" : "positive") +
                                    " decimal integer <= " + std::to_string(maximum));
    }
    return result;
}

struct cli_options {
    tq::tp5::options runtime;
    std::string      output;
    std::string      teacher_file;
    bool             help = false;
};

void usage() {
    std::cout
        << "Usage: tq-tp5-compose --model MODEL --prompt TEXT [--prompt TEXT ...] --output REPORT.json [options]\n"
           "  --prompt TEXT              Repeat 1..4 times; request ids follow argument order.\n"
           "  --mode relay|reference     Default: relay. Reference requires --windows 1.\n"
           "  --tp5-wire f32|f16         Relay payload format; default: f32. F16 requires relay mode.\n"
           "  --hc-owner-offsets LIST    Experimental HC97 owners, one 0..4 offset per window (0,0 or 0,1).\n"
           "                             Default: off. Relay mode, one request per window, full mirrored HC weights.\n"
           "  --flash-attn on|off        Explicit Flash Attention selection; default: off. No fallback.\n"
           "  --windows 1|2              Independent context windows; default: 2.\n"
           "  --predict N                Output tokens per request; default: 32.\n"
           "  --ctx-size N               Context tokens per request; default: 512; Swift requires 262144.\n"
           "  --threads N                CPU threads; default: 4.\n"
           "  --deadline N               Runtime deadline in seconds; default: 300.\n"
           "  --ignore-eos               Continue through EOS until the token budget.\n"
           "  --capture-logits           Save every full-vocabulary row as f32le plus JSONL identities.\n"
           "  --teacher-file FILE        One line per prompt, exactly predict decimal token ids per line.\n"
           "                             Only ASCII spaces separate ids; no blank lines or comments.\n"
           "  --delay-window N           Zero-based delayed consumer window (requires delay-publish-ms).\n"
           "  --delay-publish-ms N       Positive CPU publication delay in milliseconds.\n"
           "  --cancel-request N         Zero-based request id (requires cancel-after).\n"
           "  --cancel-after N           Cancel after N accepted output tokens, 1 <= N < predict.\n"
           "  --abort-window N           Inject a fatal error in this window (requires abort-channel).\n"
           "  --abort-channel N          Zero-based channel at which to inject the error.\n"
           "  --abort-epoch N            Positive matching epoch; default: 1. Requires abort-window/channel.\n"
           "  --output FILE              Required; exclusively publish complete JSON last, never overwrite.\n"
           "                             Companions: FILE.logits.f32le and FILE.logits.rows.jsonl.\n"
           "  --help                     Show help. Non-prompt options cannot be repeated.\n"
           "Output path components must be nonsymlinks; output '..' is rejected. No per-token stdout.\n";
}

void read_teacher(cli_options & result) {
    fd_owner    fd(open_input(result.teacher_file));
    struct stat info{};
    if (::fstat(fd.value, &info) != 0) {
        system_error("stat teacher file");
    }
    if (info.st_size < 0 || info.st_size > 1024 * 1024) {
        throw std::invalid_argument("teacher file exceeds the 1 MiB bound");
    }
    FILE * file = ::fdopen(fd.value, "r");
    if (!file) {
        system_error("open teacher stream");
    }
    fd.value             = -1;
    char *      buffer   = nullptr;
    std::size_t capacity = 0;
    try {
        while (true) {
            const auto count = ::getline(&buffer, &capacity, file);
            if (count < 0) {
                break;
            }
            if (result.runtime.teacher_tokens.size() == result.runtime.prompts.size()) {
                throw std::invalid_argument("teacher file has too many request lines");
            }
            std::string_view line(buffer, static_cast<std::size_t>(count));
            if (!line.empty() && line.back() == '\n') {
                line.remove_suffix(1);
            }
            std::vector<std::int32_t> tokens;
            std::size_t               begin = 0;
            while (begin < line.size()) {
                while (begin < line.size() && line[begin] == ' ') {
                    ++begin;
                }
                if (begin == line.size()) {
                    break;
                }
                auto end = line.find(' ', begin);
                if (end == std::string_view::npos) {
                    end = line.size();
                }
                if (tokens.size() == result.runtime.predict) {
                    throw std::invalid_argument("teacher row has more than predict token ids");
                }
                tokens.push_back(
                    static_cast<std::int32_t>(number(line.substr(begin, end - begin), "teacher token", true)));
                begin = end;
            }
            if (tokens.size() != result.runtime.predict) {
                throw std::invalid_argument("each teacher row must contain exactly predict token ids");
            }
            result.runtime.teacher_tokens.push_back(std::move(tokens));
        }
        if (::ferror(file)) {
            system_error("read teacher file");
        }
        if (result.runtime.teacher_tokens.size() != result.runtime.prompts.size()) {
            throw std::invalid_argument("teacher line count must equal prompt count");
        }
    } catch (...) {
        std::free(buffer);
        ::fclose(file);
        throw;
    }
    std::free(buffer);
    if (::fclose(file) != 0) {
        system_error("close teacher file");
    }
}

cli_options parse(int argc, char ** argv) {
    cli_options           result;
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key != "--prompt" && !seen.insert(key).second) {
            throw std::invalid_argument("duplicate option: " + key);
        }
        if (key == "--help") {
            result.help = true;
            continue;
        }
        if (key == "--ignore-eos") {
            result.runtime.ignore_eos = true;
            continue;
        }
        if (key == "--capture-logits") {
            result.runtime.capture_logits = true;
            continue;
        }
        static const std::set<std::string> valued = { "--model",        "--prompt",           "--mode",
                                                      "--windows",      "--predict",          "--ctx-size",
                                                      "--threads",      "--deadline",         "--teacher-file",
                                                      "--delay-window", "--delay-publish-ms", "--cancel-request",
                                                      "--cancel-after", "--abort-window",     "--abort-channel",
                                                      "--output",       "--flash-attn",       "--abort-epoch", "--tp5-wire",
                                                      "--hc-owner-offsets" };
        if (!valued.count(key)) {
            throw std::invalid_argument("unknown option: " + key);
        }
        if (++i == argc) {
            throw std::invalid_argument("missing value for " + key);
        }
        const std::string value = argv[i];
        if (key == "--model") {
            result.runtime.model = value;
        } else if (key == "--prompt") {
            result.runtime.prompts.push_back(value);
        } else if (key == "--output") {
            result.output = value;
        } else if (key == "--teacher-file") {
            result.teacher_file = value;
        } else if (key == "--mode") {
            if (value != "relay" && value != "reference") {
                throw std::invalid_argument("invalid --mode");
            }
            result.runtime.execution = value == "relay" ? tq::tp5::mode::relay : tq::tp5::mode::reference;
        } else if (key == "--tp5-wire") {
            if (value != "f32" && value != "f16") {
                throw std::invalid_argument("invalid --tp5-wire: expected f32 or f16");
            }
            result.runtime.wire = value == "f16" ? tq::tp5::wire_type::f16 : tq::tp5::wire_type::f32;
        } else if (key == "--hc-owner-offsets") {
            std::string_view remaining(value);
            while (true) {
                const auto comma = remaining.find(',');
                const auto item = remaining.substr(0, comma);
                result.runtime.hc_owner_offsets.push_back(number(item, key, true, 4));
                if (comma == std::string_view::npos) {
                    break;
                }
                remaining.remove_prefix(comma + 1);
            }
        } else if (key == "--flash-attn") {
            if (value != "on" && value != "off") {
                throw std::invalid_argument("invalid --flash-attn: expected on or off");
            }
            result.runtime.flash_attention = value == "on";
        } else if (key == "--windows") {
            result.runtime.windows = number(value, key, false, 2);
        } else if (key == "--predict") {
            result.runtime.predict = number(value, key, false, 4096);
        } else if (key == "--ctx-size") {
            result.runtime.context_per_request = number(value, key, false, 262144);
        } else if (key == "--threads") {
            result.runtime.threads = number(value, key);
        } else if (key == "--deadline") {
            result.runtime.deadline_seconds = number(value, key, false, 3600);
        } else if (key == "--delay-window") {
            result.runtime.delay_window = number(value, key, true, 1);
        } else if (key == "--delay-publish-ms") {
            result.runtime.delay_publish_ms = number(value, key, false, 10000);
        } else if (key == "--cancel-request") {
            result.runtime.cancel_request = number(value, key, true, 3);
        } else if (key == "--cancel-after") {
            result.runtime.cancel_after = number(value, key);
        } else if (key == "--abort-window") {
            result.runtime.abort_window = number(value, key, true, 1);
        } else if (key == "--abort-channel") {
            result.runtime.abort_channel = number(value, key, true, UINT64_MAX);
        } else if (key == "--abort-epoch") {
            result.runtime.abort_epoch = number(value, key, false, UINT64_MAX);
        }
    }
    if (result.help) {
        return result;
    }
    const auto & r = result.runtime;
    if (r.model.empty() || result.output.empty() || r.prompts.empty() || r.prompts.size() > 4) {
        throw std::invalid_argument("model, output and 1..4 prompts are required");
    }
    if (!seen.count("--windows")) {
        result.runtime.windows = r.execution == tq::tp5::mode::reference || r.prompts.size() == 1 ? 1 : 2;
    }
    const auto paired = [&](const char * a, const char * b) {
        if (seen.count(a) != seen.count(b)) {
            throw std::invalid_argument(std::string(a) + " requires " + b + " and vice versa");
        }
    };
    paired("--delay-window", "--delay-publish-ms");
    paired("--cancel-request", "--cancel-after");
    paired("--abort-window", "--abort-channel");
    if (seen.count("--abort-epoch") && !seen.count("--abort-window")) {
        throw std::invalid_argument("--abort-epoch requires --abort-window and --abort-channel");
    }
    if ((r.delay_publish_ms && r.delay_window >= r.windows) || r.abort_window >= static_cast<std::int32_t>(r.windows) ||
        r.cancel_request >= static_cast<std::int32_t>(r.prompts.size()) ||
        (r.cancel_request >= 0 && r.cancel_after >= r.predict)) {
        throw std::invalid_argument("injection target or cancellation point is out of range");
    }
    if (r.execution == tq::tp5::mode::reference && (r.delay_publish_ms || r.abort_window >= 0)) {
        throw std::invalid_argument("relay publication injections require relay mode");
    }
    if (!r.hc_owner_offsets.empty() &&
        (r.execution != tq::tp5::mode::relay || r.hc_owner_offsets.size() != r.windows ||
         r.prompts.size() != r.windows)) {
        throw std::invalid_argument("--hc-owner-offsets requires relay mode, one offset and one request per window");
    }
    if (seen.count("--teacher-file")) {
        if (result.teacher_file.empty()) {
            throw std::invalid_argument("empty teacher path");
        }
        read_teacher(result);
    }
    fd_owner model(open_input(r.model));
    return result;
}

// All publication names share one pinned directory. linkat is the atomic,
// no-replace commit; the report is committed only after both logits companions.
struct publication {
    std::string              report;
    fd_owner                 directory;
    std::string              lock;
    struct stat              lock_stat{};
    bool                     locked = false;
    std::vector<std::string> temporary;

    explicit publication(const cli_options & options) :
        directory(open_parent(options.output, report)),
        lock(report + ".lock") {
        const long name_max = ::fpathconf(directory.value, _PC_NAME_MAX);
        if (name_max < 0 || report.size() + std::strlen(".logits.rows.jsonl") > static_cast<std::size_t>(name_max)) {
            throw std::invalid_argument("output basename exceeds filesystem publication-name limit");
        }
        for (const auto & name : { report, report + ".logits.f32le", report + ".logits.rows.jsonl" }) {
            struct stat st{};
            if (::fstatat(directory.value, name.c_str(), &st, AT_SYMLINK_NOFOLLOW) == 0) {
                throw std::invalid_argument("output already exists: " + name);
            }
            if (errno != ENOENT) {
                system_error("check output " + name);
            }
        }
        fd_owner fd(
            ::openat(directory.value, lock.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
        if (fd.value < 0) {
            system_error("reserve output " + lock);
        }
        locked = true;
        if (::fstat(fd.value, &lock_stat) != 0) {
            system_error("stat output lock");
        }
    }

    ~publication() {
        for (const auto & name : temporary) {
            if (!name.empty()) {
                ::unlinkat(directory.value, name.c_str(), 0);
            }
        }
        struct stat st{};
        if (locked && ::fstatat(directory.value, lock.c_str(), &st, AT_SYMLINK_NOFOLLOW) == 0 &&
            st.st_dev == lock_stat.st_dev && st.st_ino == lock_stat.st_ino) {
            ::unlinkat(directory.value, lock.c_str(), 0);
        }
    }

    std::pair<int, std::string> create() {
        const std::string name = ".tp5.tmp." + std::to_string(::getpid()) + "." + std::to_string(temporary.size());
        temporary.push_back(name);
        const int fd =
            ::openat(directory.value, name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (fd < 0) {
            temporary.pop_back();
            system_error("create output temporary");
        }
        return { fd, name };
    }

    void publish(const std::string & temporary_name, const std::string & final_name) {
        if (::linkat(directory.value, temporary_name.c_str(), directory.value, final_name.c_str(), 0) != 0) {
            system_error("exclusively publish " + final_name);
        }
        if (::unlinkat(directory.value, temporary_name.c_str(), 0) != 0) {
            system_error("remove published temporary");
        }
        for (auto & name : temporary) {
            if (name == temporary_name) {
                name.clear();
                break;
            }
        }
        if (::fsync(directory.value) != 0) {
            system_error("sync output directory");
        }
    }
};

struct output_file {
    FILE *      file = nullptr;
    std::string temporary;

    explicit output_file(publication & owner) {
        auto opened = owner.create();
        temporary   = std::move(opened.second);
        file        = ::fdopen(opened.first, "wb");
        if (!file) {
            ::close(opened.first);
            system_error("open output stream");
        }
    }

    ~output_file() {
        if (file) {
            ::fclose(file);
        }
    }

    void bytes(const void * data, std::size_t count) {
        if (count && ::fwrite(data, 1, count, file) != count) {
            system_error("write output");
        }
    }

    void text(std::string_view value) { bytes(value.data(), value.size()); }

    void integer(std::uint64_t value) {
        char       buffer[32];
        const auto converted = std::to_chars(buffer, buffer + sizeof(buffer), value);
        text({ buffer, static_cast<std::size_t>(converted.ptr - buffer) });
    }

    void signed_integer(std::int64_t value) {
        if (value < 0) {
            text("-");
            integer(static_cast<std::uint64_t>(-(value + 1)) + 1);
        } else {
            integer(value);
        }
    }

    void boolean(bool value) { text(value ? "true" : "false"); }

    void quote(std::string_view value) {
        text("\"");
        const char hex[] = "0123456789abcdef";
        for (std::size_t i = 0; i < value.size();) {
            const auto c = static_cast<unsigned char>(value[i]);
            if (c < 0x80) {
                if (c == '"' || c == '\\') {
                    text("\\");
                    bytes(&value[i], 1);
                } else if (c < 0x20) {
                    const char escaped[] = { '\\', 'u', '0', '0', hex[c >> 4], hex[c & 15] };
                    bytes(escaped, 6);
                } else {
                    bytes(&value[i], 1);
                }
                ++i;
                continue;
            }
            const std::size_t length = c >= 0xc2 && c <= 0xdf ? 2 :
                                       c >= 0xe0 && c <= 0xef ? 3 :
                                       c >= 0xf0 && c <= 0xf4 ? 4 :
                                                                0;
            bool              valid  = length && i + length <= value.size();
            std::uint32_t     scalar = length == 2 ? c & 31u : length == 3 ? c & 15u : c & 7u;
            for (std::size_t j = 1; valid && j < length; ++j) {
                const auto tail = static_cast<unsigned char>(value[i + j]);
                valid           = (tail & 0xc0) == 0x80;
                scalar          = (scalar << 6) | (tail & 63u);
            }
            valid = valid &&
                    scalar >= (length == 2 ? 0x80u :
                               length == 3 ? 0x800u :
                                             0x10000u) &&
                    scalar <= 0x10ffff && !(scalar >= 0xd800 && scalar <= 0xdfff);
            if (valid) {
                bytes(value.data() + i, length);
                i += length;
            } else {
                text("\\ufffd");
                ++i;
            }
        }
        text("\"");
    }

    void key(std::string_view value) {
        quote(value);
        text(":");
    }

    void finish() {
        if (::fflush(file) != 0 || ::fsync(::fileno(file)) != 0) {
            system_error("sync output");
        }
        FILE * closing = file;
        file           = nullptr;
        if (::fclose(closing) != 0) {
            system_error("close output");
        }
    }
};

template <class T> void integer_array(output_file & out, const T & values) {
    out.text("[");
    bool comma = false;
    for (const auto value : values) {
        if (comma) {
            out.text(",");
        }
        comma = true;
        out.signed_integer(value);
    }
    out.text("]");
}

template <std::size_t N> void unsigned_array(output_file & out, const std::array<std::uint64_t, N> & values) {
    out.text("[");
    for (std::size_t i = 0; i < N; ++i) {
        if (i) {
            out.text(",");
        }
        out.integer(values[i]);
    }
    out.text("]");
}

void token_fields(output_file & out, const tq::tp5::token_record & token) {
    out.key("input_token");
    out.signed_integer(token.input_token);
    out.text(",");
    out.key("sampled_token");
    out.signed_integer(token.sampled_token);
    out.text(",");
    out.key("emitted_token");
    out.signed_integer(token.emitted_token);
    out.text(",");
    out.key("input_position");
    out.integer(token.input_position);
    out.text(",");
    out.key("committed_ns");
    out.integer(token.committed_ns);
}

std::uint64_t write_logits(publication & owner, const tq::tp5::run_result & result) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "logits require IEEE binary32");
    output_file                      binary(owner);
    output_file                      rows(owner);
    std::uint64_t                    row = 0;
    std::array<unsigned char, 16384> buffer{};
    for (const auto & request : result.requests) {
        for (std::size_t index = 0; index < request.tokens.size(); ++index) {
            const auto & token = request.tokens[index];
            if (token.logits.size() != result.vocabulary || !result.vocabulary) {
                throw std::runtime_error("runtime omitted a full-vocabulary captured logits row");
            }
            if (row >= UINT64_MAX / 4 / result.vocabulary) {
                throw std::overflow_error("logits file offset overflow");
            }
            rows.text("{\"schema_version\":1,\"row\":");
            rows.integer(row);
            rows.text(",\"request_id\":");
            rows.integer(request.id);
            rows.text(",\"window_id\":");
            rows.integer(request.window);
            rows.text(",\"token_index\":");
            rows.integer(index);
            rows.text(",");
            token_fields(rows, token);
            rows.text(",\"vocabulary\":");
            rows.integer(result.vocabulary);
            rows.text(",\"offset_bytes\":");
            rows.integer(row * result.vocabulary * 4);
            rows.text(",\"count\":");
            rows.integer(result.vocabulary);
            rows.text(",\"dtype\":\"f32le\"}\n");
            for (std::size_t start = 0; start < token.logits.size();) {
                const auto count = std::min(buffer.size() / 4, token.logits.size() - start);
                for (std::size_t i = 0; i < count; ++i) {
                    std::uint32_t bits;
                    std::memcpy(&bits, &token.logits[start + i], 4);
                    for (unsigned shift = 0; shift < 4; ++shift) {
                        buffer[i * 4 + shift] = static_cast<unsigned char>(bits >> (shift * 8));
                    }
                }
                binary.bytes(buffer.data(), count * 4);
                start += count;
            }
            ++row;
        }
    }
    binary.finish();
    rows.finish();
    owner.publish(binary.temporary, owner.report + ".logits.f32le");
    owner.publish(rows.temporary, owner.report + ".logits.rows.jsonl");
    return row;
}

void write_report(publication &               owner,
                  const cli_options &         cli,
                  const tq::tp5::run_result & result,
                  std::uint64_t               rows) {
    output_file  out(owner);
    const auto & opts = cli.runtime;
    out.text("{\"schema_version\":1,\"status\":\"success\",\"options\":{");
    out.key("model");
    out.quote(opts.model);
    out.text(",\"output\":");
    out.quote(cli.output);
    out.text(",\"prompts\":[");
    for (std::size_t i = 0; i < opts.prompts.size(); ++i) {
        if (i) {
            out.text(",");
        }
        out.quote(opts.prompts[i]);
    }
    out.text("],\"teacher_file\":");
    out.quote(cli.teacher_file);
    out.text(",\"teacher_tokens\":[");
    for (std::size_t i = 0; i < opts.teacher_tokens.size(); ++i) {
        if (i) {
            out.text(",");
        }
        integer_array(out, opts.teacher_tokens[i]);
    }
    out.text("],\"mode\":");
    out.quote(opts.execution == tq::tp5::mode::relay ? "relay" : "reference");
    out.text(",\"tp5_wire\":");
    out.quote(opts.wire == tq::tp5::wire_type::f16 ? "f16" : "f32");
    out.text(",\"hc_owner_offsets\":[");
    for (std::size_t i = 0; i < opts.hc_owner_offsets.size(); ++i) {
        if (i) out.text(",");
        out.integer(opts.hc_owner_offsets[i]);
    }
    out.text("]");
#define OPTION_U(name)           \
    out.text(",\"" #name "\":"); \
    out.integer(opts.name)
    OPTION_U(windows);
    OPTION_U(predict);
    OPTION_U(context_per_request);
    OPTION_U(threads);
    OPTION_U(deadline_seconds);
    OPTION_U(delay_window);
    OPTION_U(delay_publish_ms);
    OPTION_U(cancel_after);
    OPTION_U(abort_channel);
    OPTION_U(abort_epoch);
#undef OPTION_U
    out.text(",\"cancel_request\":");
    out.signed_integer(opts.cancel_request);
    out.text(",\"abort_window\":");
    out.signed_integer(opts.abort_window);
    out.text(",\"capture_logits\":");
    out.boolean(opts.capture_logits);
    out.text(",\"ignore_eos\":");
    out.boolean(opts.ignore_eos);
    out.text(",\"flash_attention\":");
    out.boolean(opts.flash_attention);
    out.text("},\"run\":{");
    out.key("execution_mode");
    out.quote(result.execution_mode);
    out.text(",\"numeric_policy\":");
    out.quote(result.numeric_policy);
    out.text(",\"wire_format\":");
    out.quote(result.wire_format);
    out.text(",\"hc_weight_layout\":");
    out.quote(result.hc_weight_layout);
    out.text(",\"hc_owner_offsets\":[");
    for (std::size_t i = 0; i < result.hc_owner_offsets.size(); ++i) {
        if (i) out.text(",");
        out.integer(result.hc_owner_offsets[i]);
    }
    out.text("]");
    out.text(",\"wire_byte_scope\":");
    out.quote("valid payload bytes; protocol counters include all five ready/ack facts; packed input and CPU read bytes include computed ranks only; not PCIe or cache-line measurements");
    out.text(",\"flash_attention\":");
    out.boolean(result.flash_attention);
    out.text(",\"model_description\":");
    out.quote(result.model_description);
#define RUN_U(name)              \
    out.text(",\"" #name "\":"); \
    out.integer(result.name)
    RUN_U(vocabulary);
    RUN_U(started_ns);
    RUN_U(finished_ns);
    RUN_U(max_simultaneous_windows);
    RUN_U(reserved_tokens_after_close);
    RUN_U(active_requests_after_close);
#undef RUN_U
    out.text("},\"windows\":[");
    for (std::size_t i = 0; i < result.windows.size(); ++i) {
        if (i) {
            out.text(",");
        }
        const auto & window = result.windows[i];
        out.text("{\"id\":");
        out.integer(window.id);
        out.text(",\"hc_owner_offset\":");
        out.signed_integer(window.hc_owner_offset);
#define WINDOW_U(name)           \
    out.text(",\"" #name "\":"); \
    out.integer(window.name)
        WINDOW_U(context_tokens);
        WINDOW_U(sequence_tokens);
        WINDOW_U(kv_capacity_tokens);
        WINDOW_U(decode_calls);
        WINDOW_U(decode_submit_ns);
        WINDOW_U(native_wait_ns);
        WINDOW_U(result_commit_ns);
        WINDOW_U(input_rows);
        WINDOW_U(epochs);
        WINDOW_U(channels);
        WINDOW_U(cpu_reductions);
        WINDOW_U(input_retirements);
        WINDOW_U(output_retirements);
        WINDOW_U(cpu_input_payload_bytes);
        WINDOW_U(cpu_output_payload_bytes);
        WINDOW_U(first_submit_ns);
        WINDOW_U(last_finish_ns);
        WINDOW_U(hip_runtime_version);
#undef WINDOW_U
#define WINDOW_A(name)           \
    out.text(",\"" #name "\":"); \
    unsigned_array(out, window.name)
        WINDOW_A(rank_graphs);
        WINDOW_A(rank_compute_nodes);
        WINDOW_A(rank_relay_nodes);
        WINDOW_A(rank_published_bytes);
        WINDOW_A(rank_consumed_bytes);
        WINDOW_A(rank_input_packed_payload_bytes);
        WINDOW_A(rank_output_consumed_payload_bytes);
#undef WINDOW_A
        out.text(",\"pci_bdf\":[");
        for (std::size_t rank = 0; rank < window.pci_bdf.size(); ++rank) {
            if (rank) {
                out.text(",");
            }
            out.quote(window.pci_bdf[rank]);
        }
        out.text("],\"native_drained\":");
        out.boolean(window.native_drained);
        out.text("}");
    }
    out.text("],\"requests\":[");
    std::uint64_t row = 0;
    for (std::size_t i = 0; i < result.requests.size(); ++i) {
        if (i) {
            out.text(",");
        }
        const auto & request = result.requests[i];
        out.text("{\"id\":");
        out.integer(request.id);
        out.text(",\"window\":");
        out.integer(request.window);
        out.text(",\"prompt_tokens\":");
        integer_array(out, request.prompt_tokens);
        out.text(",\"text\":");
        out.quote(request.text);
        out.text(",\"stop_reason\":");
        out.quote(request.stop_reason);
        out.text(",\"admitted_ns\":");
        out.integer(request.admitted_ns);
        out.text(",\"finished_ns\":");
        out.integer(request.finished_ns);
        out.text(",\"tokens\":[");
        for (std::size_t j = 0; j < request.tokens.size(); ++j) {
            if (j) {
                out.text(",");
            }
            out.text("{");
            token_fields(out, request.tokens[j]);
            out.text(",\"logit_row\":");
            if (opts.capture_logits) {
                out.integer(row++);
            } else {
                out.text("null");
            }
            out.text("}");
        }
        out.text("]}");
    }
    out.text("],\"logits\":");
    if (opts.capture_logits) {
        out.text("{\"dtype\":\"f32le\",\"binary\":");
        out.quote(owner.report + ".logits.f32le");
        out.text(",\"rows_file\":");
        out.quote(owner.report + ".logits.rows.jsonl");
        out.text(",\"row_order\":\"request-major/token-index\",\"rows\":");
        out.integer(rows);
        out.text(",\"vocabulary\":");
        out.integer(result.vocabulary);
        out.text(",\"bytes\":");
        out.integer(rows * result.vocabulary * 4);
        out.text("}");
    } else {
        out.text("null");
    }
    out.text("}\n");
    out.finish();
    owner.publish(out.temporary, owner.report);
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto cli = parse(argc, argv);
        if (cli.help) {
            usage();
            return 0;
        }
        publication output(cli);  // Validate/reserve all output paths before device work.
        const auto  result = tq::tp5::run(cli.runtime);
        if (cli.runtime.abort_window >= 0) {
            throw std::runtime_error("requested abort injection did not fail");
        }
        const auto rows = cli.runtime.capture_logits ? write_logits(output, result) : 0;
        write_report(output, cli, result, rows);
        std::uint64_t tokens = 0;
        for (const auto & request : result.requests) {
            tokens += request.tokens.size();
        }
        std::cout << "report=" << cli.output << " requests=" << result.requests.size() << " emitted_tokens=" << tokens
                  << " logits_rows=" << rows << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "tq-tp5-compose: " << error.what() << '\n';
        return 1;
    }
}
