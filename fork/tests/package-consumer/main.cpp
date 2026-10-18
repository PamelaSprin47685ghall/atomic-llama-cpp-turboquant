#include "tq/calibration/reader.h"
#include "tq/compose/sampler.h"
#include "tq/corpus/trie.h"
#include "tq/sampling/shortlist.h"
#include "tq/text/stream.h"

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    std::vector<float> values{1.0f, 3.0f, 2.0f, -1.0f};
    auto result = tq::sampling::select(values.data(), static_cast<int32_t>(values.size()), 2);
    if (!result.usable || result.candidates.size() != 2 || result.candidates.front().id != 1) {
        throw std::runtime_error("installed sampling library failed");
    }
    tq::corpus::token_trie trie(4);
    const auto leaf = trie.append({1, 2, 3});
    auto path = trie.path_to(leaf);
    path[0] = 0;
    if (trie.path_to(leaf).front() != 1 || trie.request_count() != 1) {
        throw std::runtime_error("installed corpus library failed");
    }
    tq::compose::sampling_options options;
    options.top_k = 2;
    tq::compose::sampler sample(options);
    if (sample.sample(values.data(), static_cast<int32_t>(values.size())) != 1 ||
        sample.stats().compact_rows != 1) {
        throw std::runtime_error("installed composer failed");
    }

    tq::text::generation_stream text({"STOP"});
    const auto first = text.push("A\xE2\x82");
    const auto second = text.push("\xAC" "ST");
    const auto last = text.push("OPdiscarded");
    if (first.text != "A" || first.offset != 0 || first.finished ||
        second.text != "\xE2\x82\xAC" || second.offset != 1 || second.finished ||
        !last.text.empty() || last.offset != 4 || !last.finished || last.stop_index != 0 ||
        !text.finished() || !text.finish().text.empty()) {
        throw std::runtime_error("installed text library failed");
    }

    // An empty WXQCAL03 file has six U32 and three U64 zero fields.
    std::string calibration_bytes("WXQCAL03");
    calibration_bytes.append(6 * 4 + 3 * 8, '\0');
    calibration_bytes.push_back('\1');
    calibration_bytes.append(7, '\0');
    std::istringstream input(calibration_bytes, std::ios::in | std::ios::binary);
    tq::calibration::reader calibration(input);
    const auto header = calibration.header();
    if (header.layers != 0 || !header.complete || header.decoded != 0 || header.total != 0 ||
        calibration.next() || calibration.next()) {
        throw std::runtime_error("installed calibration library failed");
    }
}
