#include "o200k-codec.h"

#include "ggml.h"

#include <cstdio>
#include <string>
#include <vector>

static void test_o200k_encoding() {
    wanxiangqi_o200k_codec codec;

    GGML_ASSERT(codec.encode("hello world") == std::vector<int>({24912, 2375}));
    GGML_ASSERT(codec.encode("foo_bar123") == std::vector<int>({16660, 31828, 7633}));
}

static void test_parallel_encoding_matches_serial() {
    wanxiangqi_o200k_codec codec;
    std::string text;

    for (int i = 0; i < 512; ++i) {
        text += "template <typename T> auto worker_" + std::to_string(i) + "(T value) { return value + \"\u8BA2\u5355-42\"; }\n";
        if (i % 97 == 0) {
            text += "<|endoftext|> ordinary repository text <|endofprompt|>\n";
        }
        if (i % 53 == 0) {
            text += "// slash-prefixed line\n\n  indented line\nnext_safe_line\n";
        }
    }

    const auto serial = codec.encode(text);
    GGML_ASSERT(codec.encode_parallel(text) == serial);
    GGML_ASSERT(codec.encode_parallel(text, 16) == serial);
}

int main() {
    test_o200k_encoding();
    test_parallel_encoding_matches_serial();
    std::printf("o200k codec tests passed\n");
    return 0;
}
