#include "gguf-slab.h"
#include "parallel.h"

#include <cstdio>
#include <cstring>

namespace wxq {

gguf_slab_reader::~gguf_slab_reader() {
    if (gctx) {
        gguf_free(gctx);
    }
    if (mctx) {
        ggml_free(mctx);
    }
}

bool gguf_slab_reader::open(const std::string & path) {
    gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx      = &mctx;
    gctx = gguf_init_from_file(path.c_str(), gp);
    if (!gctx) {
        fprintf(stderr, "gguf_init_from_file failed: %s\n", path.c_str());
        return false;
    }
    data_off = gguf_get_data_offset(gctx);
    f.open(path, std::ios::binary);
    return (bool) f;
}

const ggml_tensor * gguf_slab_reader::find(const std::string & name) const {
    return mctx ? ggml_get_tensor(mctx, name.c_str()) : nullptr;
}

bool gguf_slab_reader::slab(const std::string & name, int64_t expert, std::vector<float> & out,
                            int64_t & ne0, int64_t & nrows, ggml_type & type) {
    const ggml_tensor * t = find(name);
    if (!t) {
        fprintf(stderr, "missing tensor %s\n", name.c_str());
        return false;
    }
    const int64_t id = gguf_find_tensor(gctx, name.c_str());
    if (id < 0) {
        return false;
    }
    ne0   = t->ne[0];
    nrows = t->ne[1];
    type  = t->type;

    const size_t row_bytes  = ggml_row_size(type, ne0);
    const size_t slab_bytes = row_bytes * (size_t) nrows;

    std::vector<uint8_t> raw(slab_bytes);
    f.seekg((std::streamoff) (data_off + gguf_get_tensor_offset(gctx, id) + (size_t) expert*slab_bytes));
    f.read((char *) raw.data(), (std::streamsize) slab_bytes);
    if (!f) {
        fprintf(stderr, "short read on %s\n", name.c_str());
        return false;
    }

    out.resize((size_t) ne0 * (size_t) nrows);
    const auto * tr = ggml_get_type_traits(type);
    if (type == GGML_TYPE_F32) {
        std::memcpy(out.data(), raw.data(), slab_bytes);
    } else if (tr->to_float) {
        parallel_for(nrows, [&](int64_t th, int64_t step) {
            for (int64_t r = th; r < nrows; r += step) {
                tr->to_float(raw.data() + r*row_bytes, out.data() + r*ne0, ne0);
            }
        });
    } else {
        fprintf(stderr, "no to_float for %s\n", ggml_type_name(type));
        return false;
    }
    return true;
}

bool gguf_slab_reader::slab(const std::string & name, int64_t expert, std::vector<float> & out,
                            int64_t & ne0, int64_t & nrows) {
    ggml_type type = GGML_TYPE_F32;
    return slab(name, expert, out, ne0, nrows, type);
}

} // namespace wxq
