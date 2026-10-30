#pragma once

#include <llama.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tq::rerot_compose {

inline std::string json_string(std::string_view value) {
    std::string output = "\"";
    constexpr char hex[] = "0123456789abcdef";
    for (std::size_t index = 0; index < value.size();) {
        const auto c = static_cast<unsigned char>(value[index]);
        if (c >= 0x80) {
            const std::size_t size = c >= 0xc2 && c <= 0xdf ? 2 :
                                     c >= 0xe0 && c <= 0xef ? 3 :
                                     c >= 0xf0 && c <= 0xf4 ? 4 : 0;
            bool valid = size != 0 && size <= value.size() - index;
            for (std::size_t j = 1; valid && j < size; ++j) {
                const auto next = static_cast<unsigned char>(value[index + j]);
                valid = next >= 0x80 && next <= 0xbf;
                if (j == 1) {
                    valid = valid && !(c == 0xe0 && next < 0xa0) && !(c == 0xed && next >= 0xa0) &&
                                      !(c == 0xf0 && next < 0x90) && !(c == 0xf4 && next >= 0x90);
                }
            }
            if (valid) { output.append(value.substr(index, size)); index += size; }
            else { output += "\\ufffd"; ++index; }
            continue;
        }
        if (c == '\\' || c == '"') { output += '\\'; output += char(c); }
        else if (c < 0x20) { output += "\\u00"; output += hex[c >> 4]; output += hex[c & 15]; }
        else { output += char(c); }
        ++index;
    }
    output += '"';
    return output;
}

inline std::vector<llama_token> tokenize(const llama_vocab * vocab,
                                        const std::string & text, bool special) {
    if (text.size() > std::size_t(INT32_MAX)) { throw std::invalid_argument("text exceeds tokenizer capacity"); }
    const auto needed = llama_tokenize(vocab, text.data(), std::int32_t(text.size()), nullptr, 0, special, true);
    if (needed == INT32_MIN) { throw std::runtime_error("tokenizer size overflow"); }
    std::vector<llama_token> result(std::size_t(needed < 0 ? -needed : needed));
    const auto count = llama_tokenize(vocab, text.data(), std::int32_t(text.size()),
                                     result.data(), std::int32_t(result.size()), special, true);
    if (count < 0 || std::size_t(count) > result.size()) { throw std::runtime_error("tokenizer size changed"); }
    result.resize(std::size_t(count));
    return result;
}

inline std::string piece(const llama_vocab * vocab, llama_token token) {
    char small[128];
    const auto count = llama_token_to_piece(vocab, token, small, sizeof(small), 0, false);
    if (count >= 0) { return std::string(small, std::size_t(count)); }
    if (count == INT32_MIN) { throw std::runtime_error("token piece size overflow"); }
    std::string result(std::size_t(-count), '\0');
    const auto actual = llama_token_to_piece(vocab, token, result.data(), std::int32_t(result.size()), 0, false);
    if (actual < 0 || std::size_t(actual) > result.size()) { throw std::runtime_error("token piece size changed"); }
    result.resize(std::size_t(actual));
    return result;
}

class framing {
public:
    framing(const llama_model * model, std::string override_template, std::string question)
        : vocab_(llama_model_get_vocab(model)), question_(std::move(question)) {
        if (override_template.empty()) {
            const char * native = llama_model_chat_template(model, nullptr);
            if (!native) { throw std::invalid_argument("model has no chat template; supply --chat-template"); }
            template_ = native;
        } else { template_ = std::move(override_template); }
        base_text_ = render({{"user", question_.c_str()}});
        base_ = tokenize(vocab_, base_text_, true);
        if (base_.empty()) { throw std::invalid_argument("chat template produced an empty base"); }
    }

    const std::vector<llama_token> & base() const noexcept { return base_; }

    // A complete native close + user handoff + assistant open. It depends only
    // on this stage's instruction, not on the identity or text of a peer.
    std::vector<llama_token> frame(const std::string & instruction) const {
        const auto full = render({{"user", question_.c_str()}, {"assistant", ""},
                                  {"user", instruction.c_str()}});
        const auto tokens = tokenize(vocab_, full, true);
        if (tokens.size() <= base_.size() ||
            !std::equal(base_.begin(), base_.end(), tokens.begin())) {
            throw std::invalid_argument("chat template cannot preserve the base token prefix for stage framing");
        }
        return std::vector<llama_token>(tokens.begin() + std::ptrdiff_t(base_.size()), tokens.end());
    }

private:
    std::string render(const std::vector<llama_chat_message> & messages) const {
        auto count = llama_chat_apply_template(template_.c_str(), messages.data(), messages.size(), true, nullptr, 0);
        if (count <= 0) { throw std::invalid_argument("chat template is not supported by the public llama renderer"); }
        std::string output(std::size_t(count), '\0');
        const auto actual = llama_chat_apply_template(template_.c_str(), messages.data(), messages.size(), true,
                                                     output.data(), std::int32_t(output.size()));
        if (actual <= 0 || actual > count) { throw std::runtime_error("chat template size changed"); }
        output.resize(std::size_t(actual));
        return output;
    }

    const llama_vocab * vocab_;
    std::string template_;
    std::string question_;
    std::string base_text_;
    std::vector<llama_token> base_;
};

} // namespace tq::rerot_compose
