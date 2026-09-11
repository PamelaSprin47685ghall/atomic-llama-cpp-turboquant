// Read one expert slab out of a GGUF file and dequantise it.
//
// The offline probes never run the model, so they never build a backend or
// allocate the graph. What they need is the raw weights: `no_alloc` metadata plus
// a plain file handle, seeking to `data_offset + tensor_offset + expert*slab`.
//
// Expert tensors are stored as [ne0, ne1, n_expert], so expert `e` is a
// contiguous slab of ne1 rows. Non-expert tensors are read with expert 0.
//
// Bias note for every caller: source weights in the shipped checkpoints are
// already iq2_xs/iq3_xxs, so what comes back carries broadband quantisation
// noise on top of the true weights. Noise fills in small singular values and
// raises entropy, so low-rank and rate measurements taken here are pessimistic.

#pragma once

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace wxq {

struct gguf_slab_reader {
    gguf_context * gctx = nullptr;
    ggml_context * mctx = nullptr;
    std::ifstream  f;
    size_t         data_off = 0;

    gguf_slab_reader() = default;
    gguf_slab_reader(const gguf_slab_reader &) = delete;
    gguf_slab_reader & operator=(const gguf_slab_reader &) = delete;
    ~gguf_slab_reader();

    // Diagnostics go to stderr and false is returned; these are single-purpose
    // offline tools whose only recovery is to stop.
    bool open(const std::string & path);

    // Tensor metadata, or nullptr. The pointer is owned by this reader.
    const ggml_tensor * find(const std::string & name) const;

    // Dequantise slab `expert` of `name` into `out` as nrows x ne0 row-major f32.
    bool slab(const std::string & name, int64_t expert, std::vector<float> & out,
              int64_t & ne0, int64_t & nrows, ggml_type & type);

    // Same, for callers that do not care what the source type was.
    bool slab(const std::string & name, int64_t expert, std::vector<float> & out,
              int64_t & ne0, int64_t & nrows);
};

} // namespace wxq
