#include "o200k-codec.h"
#include "repetition-corpus.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::size_t half_life = 256;

std::string read_file(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to read " + path.string());
    }

    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

struct affine_replay {
    std::vector<double> offsets;
    long double coefficient_sum;
    long double offset_sum;
};

struct envelope_stats {
    double minimum;
    double maximum;
};

affine_replay replay_affine(const std::vector<int> & tokens, double lambda) {
    if (tokens.empty()) {
        throw std::runtime_error("repository corpus has no o200k tokens");
    }

    std::unordered_map<int, int64_t> last_seen;
    std::vector<double> offsets;
    offsets.reserve(tokens.size());

    double coefficient = 1.0;
    double offset = 0.0;
    long double coefficient_sum = 0.0;
    long double offset_sum = 0.0;

    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const int64_t step = static_cast<int64_t>(i) + 1;
        const int token = tokens[i];
        const auto found = last_seen.find(token);
        const double replacement = found == last_seen.end()
            ? 1.0
            : 1.0 - std::pow(lambda, static_cast<double>(step - found->second));

        coefficient *= lambda;
        offset = lambda * offset + replacement;
        last_seen[token] = step;
        offsets.push_back(offset);
        coefficient_sum += coefficient;
        offset_sum += offset;
    }

    return {std::move(offsets), coefficient_sum, offset_sum};
}

double solve_normal_prior(const affine_replay & replay) {
    const long double count = static_cast<long double>(replay.offsets.size());
    const long double mean_coefficient = replay.coefficient_sum / count;
    const long double mean_offset = replay.offset_sum / count;
    if (!(mean_coefficient < 1.0L)) {
        throw std::runtime_error("invalid repetition envelope coefficient");
    }

    return static_cast<double>(mean_offset / (1.0L - mean_coefficient));
}

envelope_stats evaluate_envelope(const std::vector<double> & offsets, double lambda, double normal_prior) {
    double coefficient = 1.0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();

    for (double offset : offsets) {
        coefficient *= lambda;
        const double weighted_distinct = coefficient * normal_prior + offset;
        minimum = std::min(minimum, weighted_distinct);
        maximum = std::max(maximum, weighted_distinct);
    }

    return {minimum, maximum};
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::cerr << "usage: derive-repetition-envelope ROOT MANIFEST OUTPUT\n";
        return 2;
    }

    try {
        const fs::path root = argv[1];
        const fs::path manifest = argv[2];
        const fs::path output = argv[3];

        wanxiangqi_o200k_codec codec;
        std::ifstream paths(manifest);
        if (!paths) {
            throw std::runtime_error("failed to read corpus manifest");
        }

        std::vector<std::string> texts;
        std::string rel;
        std::size_t source_files = 0;

        while (std::getline(paths, rel)) {
            if (rel.empty() || !wanxiangqi_repetition_corpus_path(fs::path(rel))) {
                continue;
            }

            std::string text;
            try {
                text = read_file(root / fs::path(rel));
            } catch (const std::exception &) {
                continue;
            }

            if (!wanxiangqi_repetition_corpus_text(text)) {
                continue;
            }

            texts.push_back(std::move(text));
            ++source_files;
        }

        const double lambda = std::pow(2.0, -1.0 / half_life);

        std::size_t corpus_bytes = texts.empty() ? 0 : texts.size() - 1;
        for (const auto & text : texts) {
            corpus_bytes += text.size();
        }

        const unsigned int hardware_workers = std::thread::hardware_concurrency();
        const std::size_t worker_count = hardware_workers == 0 ? 1 : hardware_workers;

        std::cerr << "o200k repetition corpus: files=" << source_files
                  << " bytes=" << corpus_bytes
                  << " workers=" << worker_count << "\n";

        std::string corpus;
        corpus.reserve(corpus_bytes);
        for (std::size_t i = 0; i < texts.size(); ++i) {
            if (i != 0) {
                corpus.push_back('\n');
            }
            corpus += texts[i];
        }

        const std::vector<int> tokens = codec.encode_parallel(corpus, worker_count);
        const affine_replay replay = replay_affine(tokens, lambda);
        const double normal_prior = solve_normal_prior(replay);
        const envelope_stats envelope = evaluate_envelope(replay.offsets, lambda, normal_prior);

        std::ofstream out(output, std::ios::trunc);
        if (!out) {
            throw std::runtime_error("failed to write envelope header");
        }

        out << "#pragma once\n\n";
        out << "#include <cstddef>\n\n";
        out << "namespace wanxiangqi_repetition_envelope {\n";
        out << std::setprecision(17);
        out << "inline constexpr std::size_t vocabulary_size = " << codec.vocabulary_size() << ";\n";
        out << "inline constexpr std::size_t half_life = " << half_life << ";\n";
        out << "inline constexpr double lambda = " << lambda << ";\n";
        out << "inline constexpr double normal_weighted_distinct_count = " << normal_prior << ";\n";
        out << "inline constexpr double minimum_weighted_distinct_count = " << envelope.minimum << ";\n";
        out << "inline constexpr double maximum_weighted_distinct_count = " << envelope.maximum << ";\n";
        out << "inline constexpr std::size_t source_files = " << source_files << ";\n";
        out << "inline constexpr std::size_t corpus_tokens = " << tokens.size() << ";\n";
        out << "}\n";

        std::cerr << "o200k repetition envelope: files=" << source_files
                  << " tokens=" << tokens.size()
                  << " half_life=" << half_life
                  << " min=" << envelope.minimum
                  << " normal=" << normal_prior
                  << " max=" << envelope.maximum << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "derive-repetition-envelope: " << e.what() << "\n";
        return 1;
    }
}
