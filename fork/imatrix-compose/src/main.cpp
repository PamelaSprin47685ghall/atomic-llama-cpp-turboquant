#include <gguf.h>
#include <tq/calibration/reader.h>

#include "output-files.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <vector>

namespace {

struct options {
    std::string input;
    std::string output;
    std::string dataset;
    tq::calibration::read_limits limits;
    std::uint64_t max_output_bytes = 2ULL * 1024 * 1024 * 1024;
    bool allow_partial = false;
    bool global_mean = false;
};

void usage() {
    std::cout <<
        "Usage: tq-imatrix-compose --input CALIB.bin --output IMATRIX.gguf --dataset NAME [options]\n"
        "Convert WXQCAL03 expert moments to the installed A imatrix format.\n"
        "  --allow-partial         Accept incomplete, underfilled, or overshot ranges.\n"
        "                         Record the original counters and all reasons.\n"
        "  --unseen POLICY         reject (default) or global-mean. Imputed rows are\n"
        "                         recorded as estimates, never measured samples.\n"
        "  --max-layer-bytes N     Maximum input bytes per layer; default: 268435456.\n"
        "  --max-input-bytes N     Maximum total input bytes; default: 17179869184.\n"
        "  --max-output-bytes N    Maximum GGUF bytes; default: 2147483648.\n"
        "No input truncation, count rounding, model loading, or GPU initialization.\n";
}

std::uint64_t number(std::string_view text, const std::string & name) {
    std::uint64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !value || value > INT64_MAX) {
        throw std::invalid_argument(name + " requires a positive decimal integer <= INT64_MAX");
    }
    return value;
}

void check_text(std::string_view text) {
    if (text.empty() || text.size() > 4096) { throw std::invalid_argument("text arguments require 1 to 4096 UTF-8 bytes"); }
    for (std::size_t i = 0; i < text.size();) {
        const auto first = static_cast<unsigned char>(text[i]);
        if (first < 128) {
            if (first < 32 || first == 127) { throw std::invalid_argument("text arguments contain control bytes"); }
            ++i;
            continue;
        }
        const unsigned count = first >= 0xc2 && first <= 0xdf ? 2 : first >= 0xe0 && first <= 0xef ? 3 : first >= 0xf0 && first <= 0xf4 ? 4 : 0;
        std::uint32_t scalar = first & (count == 2 ? 31 : count == 3 ? 15 : 7);
        bool valid = count && i + count <= text.size();
        for (unsigned j = 1; valid && j < count; ++j) {
            const auto tail = static_cast<unsigned char>(text[i + j]);
            valid = (tail & 0xc0) == 0x80;
            scalar = (scalar << 6) | (tail & 63);
        }
        if (!valid || (count == 3 && scalar < 0x800) || (count == 4 && scalar < 0x10000) ||
            scalar > 0x10ffff || (scalar >= 0xd800 && scalar <= 0xdfff)) { throw std::invalid_argument("text argument is not valid UTF-8"); }
        i += count;
    }
}

options parse(int argc, char ** argv) {
    options args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--allow-partial") { args.allow_partial = true; continue; }
        if (key != "--input" && key != "--output" && key != "--dataset" && key != "--unseen" &&
            key != "--max-layer-bytes" && key != "--max-input-bytes" && key != "--max-output-bytes") {
            throw std::invalid_argument("unknown option: " + key);
        }
        if (++i == argc) { throw std::invalid_argument("missing value for " + key); }
        const std::string value = argv[i];
        if (key == "--input") { args.input = value; }
        else if (key == "--output") { args.output = value; }
        else if (key == "--dataset") { args.dataset = value; }
        else if (key == "--unseen") {
            if (value != "reject" && value != "global-mean") { throw std::invalid_argument("--unseen requires reject or global-mean"); }
            args.global_mean = value == "global-mean";
        } else if (key == "--max-layer-bytes") { args.limits.max_layer_bytes = number(value, key); }
        else if (key == "--max-input-bytes") { args.limits.max_file_bytes = number(value, key); }
        else { args.max_output_bytes = number(value, key); }
    }
    if (args.input.empty() || args.output.empty() || args.dataset.empty()) { throw std::invalid_argument("--input, --output, and --dataset are required"); }
    check_text(args.input);
    check_text(args.output);
    check_text(args.dataset);
    return args;
}

float checked_float(double value) {
    if (!std::isfinite(value) || value < 0 || value > std::numeric_limits<float>::max()) {
        throw std::invalid_argument("squared sum does not fit a finite nonnegative F32");
    }
    const float converted = static_cast<float>(value);
    if (value != 0 && converted == 0) { throw std::invalid_argument("nonzero squared sum underflows F32"); }
    return converted;
}

void put_float(unsigned char * output, float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    for (unsigned i = 0; i < 4; ++i) { output[i] = static_cast<unsigned char>(bits >> (8 * i)); }
}

struct moments {
    const std::vector<double> & sums;
    const std::vector<std::uint64_t> & counts;
    std::uint32_t width;
    std::vector<double> mean;
};

moments prepare_moments(const std::vector<double> & sums, const std::vector<std::uint64_t> & counts,
                        std::uint32_t width, std::uint32_t layer, bool allow_mean, std::vector<std::uint64_t> & imputed) {
    moments result{sums, counts, width, {}};
    std::uint64_t total = 0;
    bool unseen = false;
    for (std::size_t expert = 0; expert < counts.size(); ++expert) {
        const auto count = counts[expert];
        if (count > INT32_MAX || double(static_cast<float>(count)) != double(count)) {
            throw std::invalid_argument("sample count cannot be represented exactly as F32 within INT32_MAX");
        }
        if (count) {
            if (count > UINT64_MAX - total) { throw std::invalid_argument("total expert sample count overflows"); }
            total += count;
            continue;
        }
        unseen = true;
        if (!allow_mean) { throw std::invalid_argument("layer " + std::to_string(layer) + " has unseen expert " + std::to_string(expert) + "; use --unseen global-mean only to request an estimate"); }
        if (imputed.size() >= 65536) { throw std::invalid_argument("imputation metadata exceeds 65536 expert records"); }
        imputed.push_back((std::uint64_t(layer) << 32) | expert);
    }
    if (!total) { throw std::invalid_argument("cannot estimate a layer with no observed experts"); }
    if (unseen) {
        result.mean.assign(width, 0);
        for (std::size_t expert = 0; expert < counts.size(); ++expert) {
            if (!counts[expert]) { continue; }
            for (std::uint32_t column = 0; column < width; ++column) {
                result.mean[column] += sums[expert * width + column] / double(total);
            }
        }
        for (const auto value : result.mean) { checked_float(value); }
    }
    return result;
}

class imatrix_writer {
public:
    imatrix_writer(tq_imatrix::private_file & spool, std::uint32_t layers, std::uint64_t maximum)
        : spool_(spool), maximum_(maximum), metadata_(gguf_init_empty(), gguf_free), tensors_(nullptr, ggml_free) {
        const auto count = std::size_t(layers) * 6;
        if (!layers || count > SIZE_MAX / ggml_tensor_overhead()) { throw std::invalid_argument("invalid output tensor count"); }
        tensors_.reset(ggml_init({count * ggml_tensor_overhead(), nullptr, true}));
        if (!metadata_ || !tensors_) { throw std::runtime_error("cannot allocate GGUF tensor metadata"); }
    }
    imatrix_writer(const imatrix_writer &) = delete;
    imatrix_writer & operator=(const imatrix_writer &) = delete;
    ~imatrix_writer() { metadata_.reset(); tensors_.reset(); }

    gguf_context * metadata() { return metadata_.get(); }
    std::uint64_t payload_bytes() const { return bytes_; }

    void append(const std::string & base, const moments & data) {
        begin_tensor(base + ".in_sum2", data.width, data.counts.size());
        std::array<unsigned char, 16384> buffer;
        for (std::size_t first = 0; first < data.sums.size();) {
            const auto count = std::min(data.sums.size() - first, buffer.size() / 4);
            for (std::size_t i = 0; i < count; ++i) {
                const auto index = first + i;
                const double value = data.counts[index / data.width] ? data.sums[index] : data.mean[index % data.width];
                const auto count_value = std::max<std::uint64_t>(data.counts[index / data.width], 1);
                const float converted = checked_float(value);
                const float normalized = converted / static_cast<float>(count_value);
                if (converted != 0 && normalized == 0) { throw std::invalid_argument("nonzero importance underflows after A's F32 count normalization"); }
                put_float(buffer.data() + 4 * i, converted);
            }
            write(buffer.data(), count * 4);
            first += count;
        }
        pad();
        begin_tensor(base + ".counts", 1, data.counts.size());
        for (std::size_t first = 0; first < data.counts.size();) {
            const auto count = std::min(data.counts.size() - first, buffer.size() / 4);
            for (std::size_t i = 0; i < count; ++i) { put_float(buffer.data() + 4 * i, static_cast<float>(std::max<std::uint64_t>(data.counts[first + i], 1))); }
            write(buffer.data(), count * 4);
            first += count;
        }
        pad();
    }

private:
    void begin_tensor(const std::string & name, std::uint64_t width, std::uint64_t experts) {
        if (name.size() >= GGML_MAX_NAME || !width || !experts || width > INT64_MAX || experts > INT64_MAX || width > SIZE_MAX / experts / 4) {
            throw std::invalid_argument("invalid GGUF tensor geometry");
        }
        auto * tensor = ggml_new_tensor_2d(tensors_.get(), GGML_TYPE_F32, static_cast<std::int64_t>(width), static_cast<std::int64_t>(experts));
        if (!tensor) { throw std::runtime_error("cannot create GGUF tensor metadata"); }
        ggml_set_name(tensor, name.c_str());
        gguf_add_tensor(metadata_.get(), tensor);
        const auto id = gguf_get_n_tensors(metadata_.get()) - 1;
        if (gguf_get_tensor_offset(metadata_.get(), id) != bytes_ || gguf_get_tensor_size(metadata_.get(), id) != width * experts * 4) {
            throw std::runtime_error("installed A GGUF tensor layout differs from the payload plan");
        }
    }
    void write(const void * data, std::size_t count) {
        if (count > maximum_ - bytes_) { throw std::invalid_argument("imatrix exceeds --max-output-bytes"); }
        spool_.write(data, count);
        bytes_ += count;
    }
    void pad() {
        const std::size_t alignment = gguf_get_alignment(metadata_.get());
        if (!alignment || alignment > 4096) { throw std::runtime_error("unsupported GGUF alignment"); }
        const std::array<unsigned char, 4096> zero{};
        write(zero.data(), (alignment - bytes_ % alignment) % alignment);
    }

    tq_imatrix::private_file & spool_;
    std::uint64_t maximum_;
    std::uint64_t bytes_ = 0;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors_;
};

std::vector<std::string> partial_reasons(const tq::calibration::file_header & header) {
    std::vector<std::string> reasons;
    if (!header.complete) { reasons.emplace_back("collector_reported_incomplete"); }
    if (!header.total) { reasons.emplace_back("declared_total_is_zero"); }
    else if (header.decoded < header.total) { reasons.emplace_back("decoded_below_declared_total"); }
    else if (header.decoded > header.total) { reasons.emplace_back("decoded_above_declared_total"); }
    return reasons;
}

void set_metadata(gguf_context * meta, const options & args, const tq::calibration::file_header & header,
                  const std::vector<std::string> & reasons, const std::vector<std::uint64_t> & input_estimates,
                  const std::vector<std::uint64_t> & hidden_estimates, std::uint64_t source_bytes) {
    gguf_set_val_str(meta, "general.architecture", "imatrix");
    gguf_set_val_str(meta, "general.type", "imatrix");
    const char * dataset = args.dataset.c_str();
    gguf_set_arr_str(meta, "imatrix.datasets", &dataset, 1);
    gguf_set_val_u32(meta, "imatrix.chunk_count", static_cast<std::uint32_t>(header.decoded));
    gguf_set_val_u32(meta, "imatrix.chunk_size", 1);
    gguf_set_val_str(meta, "tq.calibration.format", "WXQCAL03-little-endian");
    gguf_set_val_str(meta, "tq.calibration.source_path", args.input.c_str());
    gguf_set_val_u64(meta, "tq.calibration.source_bytes", source_bytes);
    gguf_set_val_str(meta, "tq.calibration.scope", "declared selected range; full corpus and model identity are unverified");
    gguf_set_val_bool(meta, "tq.calibration.complete", header.complete);
    gguf_set_val_u64(meta, "tq.calibration.decoded", header.decoded);
    gguf_set_val_u64(meta, "tq.calibration.total", header.total);
    gguf_set_val_u64(meta, "tq.calibration.context_resets", header.context_resets);
    gguf_set_val_u32(meta, "tq.calibration.max_context", header.max_context);
    gguf_set_val_u32(meta, "tq.calibration.activation_stride", header.activation_stride);
    gguf_set_val_u32(meta, "tq.calibration.activation_tokens", header.activation_tokens);
    gguf_set_val_u32(meta, "tq.calibration.route_stride", header.route_stride);
    gguf_set_val_u32(meta, "tq.calibration.flags", header.flags);
    gguf_set_val_bool(meta, "tq.calibration.partial_accepted", !reasons.empty());
    if (!reasons.empty()) {
        std::vector<const char *> values;
        for (const auto & reason : reasons) { values.push_back(reason.c_str()); }
        gguf_set_arr_str(meta, "tq.calibration.partial_reasons", values.data(), values.size());
    }
    gguf_set_val_str(meta, "tq.calibration.unseen_policy", args.global_mean ? "global-mean" : "reject");
    gguf_set_val_bool(meta, "tq.calibration.imputed_rows_are_measurements", false);
    gguf_set_val_str(meta, "tq.calibration.imputed_id_encoding", "uint64: layer_id in high 32 bits, expert_id in low 32 bits");
    gguf_set_val_u64(meta, "tq.calibration.imputed_input_count", input_estimates.size());
    gguf_set_val_u64(meta, "tq.calibration.imputed_hidden_count", hidden_estimates.size());
    if (!input_estimates.empty()) { gguf_set_arr_data(meta, "tq.calibration.imputed_input_ids", GGUF_TYPE_UINT64, input_estimates.data(), input_estimates.size()); }
    if (!hidden_estimates.empty()) { gguf_set_arr_data(meta, "tq.calibration.imputed_hidden_ids", GGUF_TYPE_UINT64, hidden_estimates.data(), hidden_estimates.size()); }
}

int convert(const options & args) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "imatrix requires IEEE binary32");
    const std::uint32_t endian = 1;
    if (*reinterpret_cast<const unsigned char *>(&endian) != 1) { throw std::runtime_error("this installed-A GGUF writer is qualified only on little-endian hosts"); }
    tq_imatrix::output_files files(args.output);
    std::ifstream input(args.input, std::ios::binary);
    if (!input) { throw std::runtime_error("cannot open calibration input"); }
    tq::calibration::reader source(input, args.limits);
    const auto header = source.header();
    if (!header.layers || !header.decoded || header.decoded > INT32_MAX) { throw std::invalid_argument("imatrix requires layers and 1..INT32_MAX decoded positions"); }
    if ((header.flags & (tq::calibration::EXPERT_INPUT | tq::calibration::HIDDEN)) != (tq::calibration::EXPERT_INPUT | tq::calibration::HIDDEN)) {
        throw std::invalid_argument("imatrix requires expert input and hidden statistics");
    }
    const auto reasons = partial_reasons(header);
    if (!reasons.empty() && !args.allow_partial) {
        std::string message = "calibration range is not complete:";
        for (const auto & reason : reasons) { message += " " + reason; }
        throw std::invalid_argument(message + "; --allow-partial records this limitation explicitly");
    }
    imatrix_writer writer(files.spool(), header.layers, args.max_output_bytes);
    std::vector<std::uint64_t> input_estimates;
    std::vector<std::uint64_t> hidden_estimates;
    while (auto layer = source.next()) {
        if (layer->id > INT32_MAX) { throw std::invalid_argument("layer id exceeds the installed A model naming range"); }
        const auto prefix = "blk." + std::to_string(layer->id) + ".ffn_";
        const auto input_moments = prepare_moments(layer->expert_sum2, layer->expert_counts, layer->embedding, layer->id, args.global_mean, input_estimates);
        const auto hidden_moments = prepare_moments(layer->hidden_sum2, layer->hidden_counts, layer->hidden, layer->id, args.global_mean, hidden_estimates);
        writer.append(prefix + "gate_exps.weight", input_moments);
        writer.append(prefix + "up_exps.weight", input_moments);
        writer.append(prefix + "down_exps.weight", hidden_moments);
    }
    set_metadata(writer.metadata(), args, header, reasons, input_estimates, hidden_estimates, source.bytes_read());
    const auto metadata_size = gguf_get_meta_size(writer.metadata());
    if (metadata_size > 16 * 1024 * 1024 || metadata_size > args.max_output_bytes - writer.payload_bytes()) {
        throw std::invalid_argument("GGUF metadata exceeds its 16 MiB limit or the output byte limit");
    }
    std::vector<unsigned char> metadata(metadata_size);
    gguf_get_meta_data(writer.metadata(), metadata.data());
    files.output().write(metadata.data(), metadata.size());
    files.spool().rewind();
    std::array<unsigned char, 65536> buffer;
    std::uint64_t copied = 0;
    while (const auto count = files.spool().read(buffer.data(), buffer.size())) {
        if (count > writer.payload_bytes() - copied) { throw std::runtime_error("payload spool grew unexpectedly"); }
        files.output().write(buffer.data(), count);
        copied += count;
    }
    if (copied != writer.payload_bytes()) { throw std::runtime_error("payload spool is incomplete"); }
    files.publish();
    std::cout << "{\"format\":\"GGUF-imatrix\",\"layers\":" << header.layers << ",\"tensors\":" << header.layers * 6
              << ",\"decoded\":" << header.decoded << ",\"declared_total\":" << header.total
              << ",\"source_bytes\":" << source.bytes_read() << ",\"output_bytes\":" << metadata_size + copied
              << ",\"partial_accepted\":" << (reasons.empty() ? "false" : "true")
              << ",\"imputed_input_experts\":" << input_estimates.size() << ",\"imputed_hidden_experts\":" << hidden_estimates.size()
              << ",\"imputed_rows_are_measurements\":false,\"model_identity_verified\":false}\n";
    std::cout.flush();
    if (!std::cout) { throw std::runtime_error("complete imatrix was published, but stdout summary failed"); }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") { usage(); std::cout.flush(); return std::cout ? 0 : 1; }
        return convert(parse(argc, argv));
    } catch (const std::invalid_argument & error) {
        std::cerr << "tq-imatrix-compose: " << error.what() << '\n';
        return 2;
    } catch (const std::exception & error) {
        std::cerr << "tq-imatrix-compose: " << error.what() << '\n';
        return 1;
    }
}
