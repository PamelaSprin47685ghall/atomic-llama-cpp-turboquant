#pragma once

#include "ggml-predefined.h"

// One resource definition shared by target and MTP. Separate mathematical entry
// points do not imply separate scratch allocations. This object deliberately
// contains no vector/map of shapes and no borrowed ggml graph pointers.
struct llama_predefined_session {
    explicit llama_predefined_session(const ggml_predefined_capacity & c) : capacity(c) {}
    const ggml_predefined_capacity capacity;
};
