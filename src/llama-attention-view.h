#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

// Owned graph input metadata. Logical identities are resolved after KV slot
// preparation. No physical index survives a decode call.
struct llama_attention_view_owned {
    llama_seq_id query_seq;
    llama_pos query_pos;
    llama_pos query_virtual_pos;
    std::vector<llama_attention_entry> entries;
};

struct llama_attention_span_resolved {
    llama_pos effective_pos;
    std::vector<int32_t> indices;
};

struct llama_attention_view_resolved {
    std::vector<llama_attention_span_resolved> spans;
    std::vector<int32_t> indices;
};
