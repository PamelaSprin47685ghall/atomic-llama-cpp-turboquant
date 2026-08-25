#include "o200k-codec.h"

#include "encoding.h"

#include <utility>

struct wanxiangqi_o200k_codec::impl {
    std::shared_ptr<GptEncoding> encoding = GptEncoding::get_encoding(LanguageModel::O200K_BASE);
};

wanxiangqi_o200k_codec::wanxiangqi_o200k_codec()
    : pimpl(std::make_unique<impl>()) {
}

wanxiangqi_o200k_codec::~wanxiangqi_o200k_codec() = default;

wanxiangqi_o200k_codec::wanxiangqi_o200k_codec(wanxiangqi_o200k_codec &&) noexcept = default;

wanxiangqi_o200k_codec & wanxiangqi_o200k_codec::operator=(wanxiangqi_o200k_codec &&) noexcept = default;

std::vector<int> wanxiangqi_o200k_codec::encode(const std::string & text) const {
    return pimpl->encoding->encode(text, {}, {});
}

std::size_t wanxiangqi_o200k_codec::vocabulary_size() const {
    return pimpl->encoding->get_byte_pair_token_map().size();
}
