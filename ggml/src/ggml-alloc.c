#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-impl.h"

#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX_FREE_BLOCKS 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#define AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(enum ggml_op op) {
    switch (op) {
        case GGML_OP_FILL:
        case GGML_OP_SCALE:
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_UNARY:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_SILU_BACK:
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SOFT_MAX_BACK:
            return true;

        default:
            return false;
    }
}

static size_t aligned_offset(const void * buffer, size_t offset, size_t alignment) {
    assert(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
    return offset + align;
}

// tallocr

struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer) {
    void * base = ggml_backend_buffer_get_base(buffer);
    size_t align = ggml_backend_buffer_get_alignment(buffer);

    assert(align && !(align & (align - 1))); // power of 2

    struct ggml_tallocr talloc = (struct ggml_tallocr) {
        /*.buffer    = */ buffer,
        /*.base      = */ base,
        /*.alignment = */ align,
        /*.offset    = */ aligned_offset(base, 0, align),
    };
    return talloc;
}

enum ggml_status ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor) {
    size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size = GGML_PAD(size, talloc->alignment);

    if (talloc->offset + size > ggml_backend_buffer_get_size(talloc->buffer)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, size, ggml_backend_buffer_get_size(talloc->buffer) - talloc->offset);
        GGML_ABORT("not enough space in the buffer");
    }

    void * addr = (char *)ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += size;

    assert(((uintptr_t)addr % talloc->alignment) == 0);

    return ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}

// dynamic tensor allocator

#define GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
struct buffer_address {
    int chunk;     // index of a backend buffer
    size_t offset; // local memory offset within the buffer
};

static const struct buffer_address GGML_BUFFER_ADDRESS_INVALID = { -1, SIZE_MAX };

static bool ggml_buffer_address_less(struct buffer_address a, struct buffer_address b) {
    return a.chunk != b.chunk ? a.chunk < b.chunk : a.offset < b.offset;
}

struct free_block {
    size_t offset;
    size_t size;
};

struct tallocr_chunk {
    struct free_block free_blocks[MAX_FREE_BLOCKS];
    int n_free_blocks;
    size_t max_size;
};

struct ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_size;
    struct tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    int n_chunks;

#ifdef GGML_ALLOCATOR_DEBUG
    struct {
        const struct ggml_tensor * tensor;
        struct buffer_address addr;
    } allocated_tensors[1024];
#endif
};

static void ggml_dyn_tallocr_insert_block(struct tallocr_chunk * chunk, size_t offset, size_t size) {
    GGML_ASSERT(chunk->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    int insert_pos = 0;
    while (insert_pos < chunk->n_free_blocks && chunk->free_blocks[insert_pos].offset < offset) {
        insert_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    for (int i = chunk->n_free_blocks; i > insert_pos; i--) {
        chunk->free_blocks[i] = chunk->free_blocks[i-1];
    }
    // insert the new block
    chunk->free_blocks[insert_pos].offset = offset;
    chunk->free_blocks[insert_pos].size = size;
    chunk->n_free_blocks++;
}

static void ggml_dyn_tallocr_remove_block(struct tallocr_chunk * chunk, int idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    for (int i = idx; i < chunk->n_free_blocks - 1; i++) {
        chunk->free_blocks[i] = chunk->free_blocks[i+1];
    }
    chunk->n_free_blocks--;
}

static int ggml_dyn_tallocr_new_chunk(struct ggml_dyn_tallocr * alloc, size_t min_size) {
    if (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        return -1;
    }
    struct tallocr_chunk * chunk = calloc(1, sizeof(struct tallocr_chunk));
    chunk->n_free_blocks = 1;
    chunk->free_blocks[0].offset = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_blocks[0].size = MAX(min_size, alloc->max_chunk_size);
    if (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_blocks[0].size = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    return alloc->n_chunks - 1;
}

#ifdef GGML_ALLOCATOR_DEBUG
static void add_allocated_tensor(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, const struct ggml_tensor * tensor) {
    for (int i = 0; i < 1024; i++) {
        if (alloc->allocated_tensors[i].tensor == NULL) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            return;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
static void remove_allocated_tensor(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, const struct ggml_tensor * tensor) {
    for (int i = 0; i < 1024; i++) {
        if (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.offset == addr.offset) {
            alloc->allocated_tensors[i].tensor = NULL;
            return;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#endif

static struct buffer_address ggml_dyn_tallocr_alloc(struct ggml_dyn_tallocr * alloc, size_t size, const struct ggml_tensor * tensor) {
    size = aligned_offset(NULL, size, alloc->alignment);

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

    int best_fit_chunk = -1;
    int best_fit_block = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    for (int c = 0; c < alloc->n_chunks; ++c) {
        struct tallocr_chunk * chunk = alloc->chunks[c];
        size_t best_fit_size = SIZE_MAX;
        for (int i = 0; i < chunk->n_free_blocks - 1; i++) {
            struct free_block * block = &chunk->free_blocks[i];
            max_avail = MAX(max_avail, block->size);
            if (block->size >= size && block->size <= best_fit_size) {
                best_fit_chunk = c;
                best_fit_block = i;
                best_fit_size = block->size;
            }
        }
    }

    if (best_fit_block == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t best_reuse = INT64_MIN;
        for (int c = 0; c < alloc->n_chunks; ++c) {
            struct tallocr_chunk * chunk = alloc->chunks[c];
            if (chunk->n_free_blocks > 0) {
                struct free_block * block = &chunk->free_blocks[chunk->n_free_blocks - 1];
                max_avail = MAX(max_avail, block->size);
                int64_t reuse_factor = chunk->max_size - block->offset - size;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = best_reuse < 0 && reuse_factor > best_reuse;
                bool better_fit = reuse_factor >= 0 && reuse_factor < best_reuse;
                if (block->size >= size && (better_reuse || better_fit)) {
                    best_fit_chunk = c;
                    best_fit_block = chunk->n_free_blocks - 1;
                    best_reuse = reuse_factor;
                }
            }
        }
    }

    if (best_fit_block == -1) {
        // none of the existing chunks have enough space left
        best_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, size);
        best_fit_block = 0;
    }
    if (best_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, size, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    struct tallocr_chunk * chunk = alloc->chunks[best_fit_chunk];
    struct free_block    * block = &chunk->free_blocks[best_fit_block];
    struct buffer_address  addr  = {.chunk = best_fit_chunk, .offset = block->offset };
    block->offset += size;
    block->size -= size;
    if (block->size == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_block(chunk, best_fit_block);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", best_fit_block, addr.offset, addr.chunk);

#ifdef GGML_ALLOCATOR_DEBUG
    add_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.offset + size;
    if (cur_max > chunk->max_size) {
        // sort allocated_tensors by chunk/offset
        for (int i = 0; i < 1024; i++) {
            for (int j = i + 1; j < 1024; j++) {
                if (ggml_buffer_address_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    const struct ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    struct buffer_address tmp_addr = alloc->allocated_tensors[i].addr;
                    alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
                    alloc->allocated_tensors[i].addr = alloc->allocated_tensors[j].addr;
                    alloc->allocated_tensors[j].tensor = tmp_tensor;
                    alloc->allocated_tensors[j].addr = tmp_addr;
                }
            }
        }
        GGML_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
        for (int i = 0; i < 1024; i++) {
            if (alloc->allocated_tensors[i].tensor) {
                GGML_LOG_DEBUG("%s [%d: %zx-%zx] (%.2f MB) ", alloc->allocated_tensors[i].tensor->name,
                    alloc->allocated_tensors[i].addr.chunk,
                    alloc->allocated_tensors[i].addr.offset,
                    alloc->allocated_tensors[i].addr.offset + ggml_nbytes(alloc->allocated_tensors[i].tensor),
                    ggml_nbytes(alloc->allocated_tensors[i].tensor) / 1024.0 / 1024.0);
            }
        }
        GGML_LOG_DEBUG("\n");
    }
#endif

    chunk->max_size = MAX(chunk->max_size, addr.offset + size);

    return addr;

    GGML_UNUSED(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
static void ggml_dyn_tallocr_free_bytes(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, size_t size) {
    size = aligned_offset(NULL, size, alloc->alignment);

    struct tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    for (int i = 0; i < chunk->n_free_blocks; i++) {
        struct free_block * block = &chunk->free_blocks[i];
        // check if ptr is at the end of the block
        if (block->offset + block->size == addr.offset) {
            block->size += size;
            // check if we can merge with the next block
            if (i < chunk->n_free_blocks - 1) {
                struct free_block * next = &chunk->free_blocks[i+1];
                if (block->offset + block->size == next->offset) {
                    block->size += next->size;
                    ggml_dyn_tallocr_remove_block(chunk, i+1);
                }
            }
            return;
        }
        // check if ptr is at the beginning of the block
        if (addr.offset + size == block->offset) {
            block->offset = addr.offset;
            block->size += size;
            // check if we can merge with the previous block
            if (i > 0) {
                struct free_block * prev = &chunk->free_blocks[i-1];
                if (prev->offset + prev->size == block->offset) {
                    prev->size += block->size;
                    ggml_dyn_tallocr_remove_block(chunk, i);
                }
            }
            return;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_insert_block(chunk, addr.offset, size);
}

static void ggml_dyn_tallocr_reset(struct ggml_dyn_tallocr * alloc) {
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        free(alloc->chunks[i]);
        alloc->chunks[i] = NULL;
    }
    alloc->n_chunks = 0;

#ifdef GGML_ALLOCATOR_DEBUG
    for (int i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = NULL;
    }
#endif
}

static struct ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_buffer_size) {
    struct ggml_dyn_tallocr * alloc = (struct ggml_dyn_tallocr *)malloc(sizeof(struct ggml_dyn_tallocr));

    *alloc = (struct ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_buffer_size, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {NULL},
        /*.n_chunks       = */ 0,
#ifdef GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#endif
    };

    ggml_dyn_tallocr_reset(alloc);

    return alloc;
}

static void ggml_dyn_tallocr_free(struct ggml_dyn_tallocr * alloc) {
    for (int i = 0; i < alloc->n_chunks; ++i) {
        free(alloc->chunks[i]);
    }
    free(alloc);
}

static size_t ggml_dyn_tallocr_max_size(struct ggml_dyn_tallocr * alloc, int chunk) {
    return chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_size : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

struct vbuffer {
    ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

static void ggml_vbuffer_free(struct vbuffer * buf) {
    if (buf == NULL) {
        return;
    }
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        ggml_backend_buffer_free(buf->chunks[i]);
    }
    free(buf);
}

static size_t ggml_vbuffer_chunk_size(struct vbuffer * buf, int chunk) {
    return buf->chunks[chunk] ? ggml_backend_buffer_get_size(buf->chunks[chunk]) : 0;
}

static size_t ggml_vbuffer_size(struct vbuffer * buf) {
    size_t size = 0;
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        size += ggml_backend_buffer_get_size(buf->chunks[i]);
    }
    return size;
}

static struct vbuffer * ggml_vbuffer_alloc(ggml_backend_buffer_type_t buft, const struct ggml_dyn_tallocr * talloc, enum ggml_backend_buffer_usage usage) {
    struct vbuffer * buf = (struct vbuffer *)calloc(1, sizeof(struct vbuffer));
    if (buf == NULL) {
        return NULL;
    }

    for (int n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_size = talloc->chunks[n]->max_size;
        buf->chunks[n] = ggml_backend_buft_alloc_buffer(buft, chunk_size);
        if (buf->chunks[n] == NULL) {
            ggml_vbuffer_free(buf);
            return NULL;
        }
        ggml_backend_buffer_set_usage(buf->chunks[n], usage);
    }
    return buf;
}

static void ggml_vbuffer_tensor_alloc(struct vbuffer * buf, struct ggml_tensor * tensor, struct buffer_address buf_addr) {
    void * base = ggml_backend_buffer_get_base(buf->chunks[buf_addr.chunk]);
    void * addr = (char *)base + buf_addr.offset;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

static void ggml_vbuffer_reset(struct vbuffer * buf) {
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_buffer_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

struct hash_node {
    int n_children;
    int n_views;
    int buffer_id;
    struct buffer_address addr;
    bool allocated;
};

struct tensor_alloc {
    int buffer_id;
    struct buffer_address addr;
    size_t size_max; // 0 = pre-allocated, unused, or view
    bool no_reuse; // allocation plans depend on the retained-lifetime contract
};

struct leaf_alloc {
    struct tensor_alloc leaf;
};

struct node_alloc {
    struct tensor_alloc dst;
    struct tensor_alloc src[GGML_MAX_SRC];
};

struct ggml_gallocr {
    ggml_backend_buffer_type_t * bufts; // [n_buffers]
    struct vbuffer ** buffers; // [n_buffers]
    struct ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    int n_buffers;

    struct ggml_hash_set hash_set;
    struct hash_node * hash_values; // [hash_set.size]

    struct node_alloc * node_allocs; // [n_nodes]
    int n_nodes;

    struct leaf_alloc * leaf_allocs; // [n_leafs]
    int n_leafs;

    // Identity of the plan this instance currently holds. The evidence lines
    // below print it, and an owner (llama_context) reads it back to check that
    // the plan it reserved is the plan it submits. Nothing here decides an
    // allocation: the fields are written after the plan is final.
    //
    // A "planning pass" is one ggml_gallocr_reserve_n_impl call, whatever its
    // entry point. plan_id counts passes over the lifetime of this allocator,
    // so a re-reserve of the same graph object (same graph_uid) still moves it.
    int64_t  plan_id;        // 0 = no planning pass yet
    uint64_t plan_graph_uid; // graph->uid of the planned graph, 0 when unset
    uint64_t plan_fp;        // ggml_gallocr_plan_fp of the planned graph
    int      plan_n_nodes;
    int      plan_n_leafs;
    bool     plan_entry_size; // true when the pass was a size measurement
    // Retention the held plan actually carries: 0 when the pass that built it
    // found no retained allocation, >0 otherwise, -1 while a pass is running or
    // when the audit is off. Written only at the end of an audited pass, so a
    // refused pass leaves -1 and a caller cannot read it as a clean plan.
    int      plan_retained;

    // Identity of the last plan the retention audit accepted **with retained
    // allocations**. Set only by such an audit, so it never describes a plan that
    // was refused, nor a plan with nothing to vouch for. Kept across later passes:
    // a refused or unaudited pass advances plan_id, so the two ids part on their own.
    int64_t  audit_clean_plan_id;
    uint64_t audit_clean_graph_uid;
    uint64_t audit_clean_fp;
    int      audit_clean_n_nodes;
    int      audit_clean_n_leafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)calloc(1, sizeof(struct ggml_gallocr));
    GGML_ASSERT(galloc != NULL);

    galloc->bufts = calloc(n_bufs, sizeof(ggml_backend_buffer_type_t));
    GGML_ASSERT(galloc->bufts != NULL);

    galloc->buffers = calloc(n_bufs, sizeof(struct vbuffer *));
    GGML_ASSERT(galloc->buffers != NULL);

    galloc->buf_tallocs = calloc(n_bufs, sizeof(struct ggml_dyn_tallocr *));
    GGML_ASSERT(galloc->buf_tallocs != NULL);

    for (int i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = NULL;

        // check if the same buffer type is used multiple times and reuse the same allocator
        for (int j = 0; j < i; j++) {
            if (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                break;
            }
        }

        if (galloc->buf_tallocs[i] == NULL) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
            size_t max_size = ggml_backend_buft_get_max_size(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_size);
        }
    }
    galloc->n_buffers = n_bufs;

    return galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft) {
    return ggml_gallocr_new_n(&buft, 1);
}

void ggml_gallocr_free(ggml_gallocr_t galloc) {
    if (galloc == NULL) {
        return;
    }

    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->buffers != NULL) {
            // skip if already freed
            bool freed = false;
            for (int j = 0; j < i; j++) {
                if (galloc->buffers[j] == galloc->buffers[i]) {
                    freed = true;
                    break;
                }
            }
            if (!freed) {
                ggml_vbuffer_free(galloc->buffers[i]);
            }
        }
        if (galloc->buf_tallocs != NULL) {
            // skip if already freed
            bool freed = false;
            for (int j = 0; j < i; j++) {
                if (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                    freed = true;
                    break;
                }
            }
            if (!freed) {
                ggml_dyn_tallocr_free(galloc->buf_tallocs[i]);
            }
        }
    }

    ggml_hash_set_free(&galloc->hash_set);
    free(galloc->hash_values);
    free(galloc->bufts);
    free(galloc->buffers);
    free(galloc->buf_tallocs);
    free(galloc->node_allocs);
    free(galloc->leaf_allocs);
    free(galloc);
}

typedef struct ggml_gallocr * ggml_gallocr_t;

static struct hash_node * ggml_gallocr_hash_get(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    size_t i = ggml_hash_find_or_insert(&galloc->hash_set, t);
    return &galloc->hash_values[i];
}

static bool ggml_gallocr_is_own(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    return ggml_gallocr_hash_get(galloc, t)->allocated;
}

static bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    return t->data != NULL // tensor data already set externally
        || t->buffer // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// FNV-1a over the node pointer array in graph order, folded with the node and
// leaf counts. The fingerprint says "the same node sequence was planned", not
// "the same computation": it cannot tell two runs of one builder apart. The
// graph uid carries that distinction, so both are printed and both are read
// back by the owner. Pure read, no allocation.
uint64_t ggml_gallocr_plan_fp(const struct ggml_cgraph * graph) {
    uint64_t fp = 0xcbf29ce484222325ull;
    const uint8_t * bytes = (const uint8_t *) graph->nodes;
    const size_t    len   = (size_t) graph->n_nodes * sizeof(graph->nodes[0]);
    for (size_t i = 0; i < len; i++) {
        fp ^= bytes[i];
        fp *= 0x100000001b3ull;
    }
    fp ^= (uint64_t) (uint32_t) graph->n_nodes;
    fp *= 0x100000001b3ull;
    fp ^= (uint64_t) (uint32_t) graph->n_leafs;
    fp *= 0x100000001b3ull;
    return fp;
}

// Fail-closed plan audit for explicitly retained allocations
// (GGML_TENSOR_FLAG_NO_REUSE, e.g. the multi-row MoE router scratch): a
// retained tensor must not share storage with any other allocation in the same
// buffer, otherwise the retention contract is not actually provided. The
// dyn_tallocr only hands out disjoint free blocks, so an overlap in a finished
// plan can only come from a bug; refuse the plan instead of running it.
// Off by default; GGML_TP5_ROUTER_SCRATCH_AUDIT=1 (exact) enables the check.
//
// Placement is what makes the audit able to pass: ggml_gallocr_alloc_graph_impl
// allocates every non-view NO_REUSE node before the first temporary, and
// ggml_gallocr_free_node never frees it, so the only address a retained tensor can
// share is its own in-place continuation (checked below). Without that, a retained
// node could take an address a temporary had already been freed from.
//
// An empty tensor (some ne[] is 0) is not an unverifiable neighbour: it has no
// bytes and no address range, and every backend's kernel is a no-op for it. The
// retained tensor itself is never allowed to be empty.
//
// Retention is tracked per plan (tensor_alloc.no_reuse, copied from the graph
// at reserve time) because ggml_tensor flags are sticky: an in-place child of a
// retained tensor keeps NO_REUSE set after the in-place pass. Retained tensors
// are never freed during graph allocation, so in a finished plan the only
// address a retained tensor can share is its own in-place continuation, which
// is legitimate -- and is exactly what a retained tensor with a later NO_REUSE
// consumer looks like. That shape is recognised explicitly, and every other
// overlap is a contract violation.

// Allocated extent of a planned tensor, for the retention audit only.
//
// size_max stores the unaligned logical size, because that is what
// ggml_gallocr_node_needs_realloc() compares against an unaligned node_size;
// the dyn_tallocr, however, rounds every allocation up to the buffer alignment
// (ggml_dyn_tallocr_alloc). Judge overlap on what was actually reserved, not on
// the logical bytes, hence the aligned extent here.
//
// This deliberately makes the audit STRICTER than the unaligned convention: a
// neighbour placed inside the alignment tail now counts as an overlap. That is
// the correct reading for a retention contract (a later in-place writer may
// legitimately target that tail), so do not "simplify" it back to size_max.
//
// Nothing computed here is written back: the allocator's own re-reserve
// decision keeps using the unaligned size_max. A tensor the plan recorded no
// size for returns 0 and the audit refuses the plan rather than guessing an
// extent. Whether a tensor belongs to this plan's address space at all is the
// caller's decision, made from buffer_id, not from whether data is set: a
// retained tensor can legitimately have been assigned by the plan and still
// carry data when the same plan is replayed.
static size_t ggml_gallocr_audit_extent(ggml_gallocr_t galloc, const struct tensor_alloc * alloc,
                                        const struct ggml_tensor * t) {
    if (alloc->buffer_id < 0) {
        return 0;
    }
    // A zero size_max means this plan recorded no size for the tensor, so there
    // is no reservation to describe; deriving one from the tensor would be a
    // guess about storage the plan does not own.
    if (alloc->size_max == 0) {
        return 0;
    }
    // audit, not the realloc predicate: judge against what a placement in this
    // buffer domain could occupy, which is the size rounded up to that buffer's
    // alignment (ggml_dyn_tallocr_alloc does the same for every allocation).
    const size_t alignment = galloc->buf_tallocs[alloc->buffer_id]->alignment;
    return aligned_offset(NULL, alloc->size_max, alignment);
}

static bool ggml_gallocr_same_block(const struct hash_node * hn, const struct tensor_alloc * alloc) {
    return hn->buffer_id == alloc->buffer_id && hn->addr.chunk == alloc->addr.chunk &&
            hn->addr.offset == alloc->addr.offset;
}

static bool ggml_gallocr_check_no_reuse_plan(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const char * entry) {
    // Scope of this switch, stated once: it is OFF unless the variable is exactly
    // "1". When it is on and the plan violates the retention contract (a
    // retained extent overlaps another allocation, a retained tensor is a view,
    // or a retained/neighbouring allocation is outside the plan so no extent can
    // be compared), the audit returns false. That propagates out of
    // ggml_gallocr_reserve_n (sched reserve fails, decode reports
    // ALLOC_FAILED) and out of the ggml_gallocr_reserve_n_size ASSERT. It is a
    // fail-closed diagnostic, NOT pure observation: enabling it can reject a
    // plan that would otherwise have run. With it off nothing is computed or
    // allocated beyond the existing reserve work.
    //
    // The audit also records the identity of the plan it judged (graph_uid,
    // plan_id, fp) and prints it on the evidence lines. That record is what lets
    // an owner check that the plan it submits is this plan; it changes no
    // allocation and is only written when the audit is enabled.
    const char * audit_env = getenv("GGML_TP5_ROUTER_SCRATCH_AUDIT");
    if (!audit_env || strcmp(audit_env, "1") != 0) {
        return true;
    }

    // Identity of the plan just produced by reserve_n_impl. Written here because
    // this is the only place that knows the audit ran; the fields are read back
    // through ggml_gallocr_plan_identity() and never feed an allocation
    // decision. plan_id was bumped by reserve_n_impl for this pass.
    galloc->plan_graph_uid = graph->uid;
    galloc->plan_fp        = ggml_gallocr_plan_fp(graph);
    galloc->plan_n_nodes   = graph->n_nodes;
    galloc->plan_n_leafs   = graph->n_leafs;

    size_t retained_bytes = 0;
    int    n_retained     = 0;
    int    n1_rows        = 0; // ne[1] of the retained projections, 0 when they disagree
    bool   n1_mixed       = false;
    int    n_unverifiable = 0; // retained tensors this audit cannot vouch for

    for (int i = 0; i < graph->n_nodes; i++) {
        const struct tensor_alloc * kept = &galloc->node_allocs[i].dst;
        if (!kept->no_reuse) {
            continue;
        }
        struct ggml_tensor * node = graph->nodes[i];

        // A retained view shares storage it does not own. This used to sit behind
        // the buffer_id guard below, which views never pass, so the violation
        // was silently accepted; decide it first.
        if (node->view_src != NULL) {
            GGML_LOG_ERROR("%s: retained tensor '%s' [%s] is a view, it cannot be retained\n",
                    __func__, node->name, ggml_op_desc(node));
            return false;
        }

        // Retained storage the plan does not describe cannot be proven disjoint:
        // when the plan placed no allocation for it (buffer_id < 0: its own data
        // or a view, already rejected above) or recorded no size for it, there is
        // no extent to compare, and guessing from another buffer_id would be
        // worse than admitting the gap. Refuse the plan; a retention contract
        // that is only partly verified must not run.
        if (kept->buffer_id < 0 || kept->size_max == 0) {
            n_unverifiable += 1;
            GGML_LOG_ERROR("%s: retained tensor '%s' [%s] is outside the audited plan "
                    "(buffer_id=%d size_max=%zu); its disjointness cannot be proven\n",
                    __func__, node->name, ggml_op_desc(node), kept->buffer_id, kept->size_max);
            return false;
        }

        // A retained in-place continuation: its own storage belongs to an
        // earlier retained producer. Audit the storage once, at the producer.
        bool continued = false;
        for (int j = 0; j < GGML_MAX_SRC && !continued; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            if (parent->view_src != NULL || (parent->flags & GGML_TENSOR_FLAG_NO_REUSE) == 0) {
                continue;
            }
            // the parent is retained and shares this exact block, so it is the
            // node that owns the storage; the continuation only names it
            continued = ggml_gallocr_same_block(ggml_gallocr_hash_get(galloc, parent), kept);
        }
        if (continued) {
            continue;
        }

        const size_t kept_size = ggml_gallocr_audit_extent(galloc, kept, node);
        if (kept_size == 0) {
            // Unreachable after the checks above; refuse rather than report a
            // clean plan that did not actually cover this tensor.
            GGML_LOG_ERROR("%s: retained tensor '%s' [%s] has no auditable extent\n",
                    __func__, node->name, ggml_op_desc(node));
            return false;
        }

        n_retained     += 1;
        retained_bytes += kept_size;
        if (n_retained == 1) {
            n1_rows = (int) node->ne[1];
        } else if (n1_rows != (int) node->ne[1]) {
            n1_mixed = true;
        }

        const size_t kept_lo = kept->addr.offset;
        const size_t kept_hi = kept_lo + kept_size;
        bool clash = false;

        for (int j = 0; j < graph->n_nodes && !clash; j++) {
            if (j == i) {
                continue;
            }
            const struct tensor_alloc * other = &galloc->node_allocs[j].dst;
            // Domain, not data: a tensor the plan placed in another buffer (or in
            // none at all) has no address in this plan's space, so the comparison
            // is vacuous by construction. Note that a tensor which *does* carry
            // data can still be in domain when the same plan is replayed, and
            // must stay compared.
            if (other->buffer_id < 0) {
                continue;
            }
            if (other->buffer_id != kept->buffer_id) {
                continue;
            }
            if (other->size_max == 0) {
                // In-domain with no recorded size. An empty tensor (some ne[] is 0)
                // has no bytes and no address range at all -- every backend's kernel
                // is a no-op for it -- so it cannot overlap anything and needs no
                // extent. Only a non-empty tensor with no recorded size is unknown.
                if (!ggml_is_empty(graph->nodes[j])) {
                    n_unverifiable += 1;
                }
                continue;
            }
            const size_t other_size = ggml_gallocr_audit_extent(galloc, other, graph->nodes[j]);
            if (other->buffer_id == kept->buffer_id && other->addr.chunk == kept->addr.chunk &&
                    other->addr.offset < kept_hi && kept_lo < other->addr.offset + other_size) {
                GGML_LOG_ERROR("%s: retained tensor '%s' [%s] overlaps '%s' [%s] in buffer %d\n",
                        __func__, node->name, ggml_op_desc(node),
                        graph->nodes[j]->name, ggml_op_desc(graph->nodes[j]), kept->buffer_id);
                clash = true;
            }
        }
        for (int j = 0; j < graph->n_leafs && !clash; j++) {
            const struct tensor_alloc * other = &galloc->leaf_allocs[j].leaf;
            // Weights, token and embedding buffers are placed outside this galloc
            // (buffer_id = -1): no address in this plan's space, nothing to
            // compare. Only an in-domain leaf with no recorded size is genuinely
            // unverifiable.
            if (other->buffer_id < 0 || other->buffer_id != kept->buffer_id) {
                continue;
            }
            if (other->size_max == 0) {
                // Same rule as for nodes: an empty tensor occupies nothing.
                if (!ggml_is_empty(graph->leafs[j])) {
                    n_unverifiable += 1;
                }
                continue;
            }
            const size_t other_size = ggml_gallocr_audit_extent(galloc, other, graph->leafs[j]);
            if (other->buffer_id == kept->buffer_id && other->addr.chunk == kept->addr.chunk &&
                    other->addr.offset < kept_hi && kept_lo < other->addr.offset + other_size) {
                GGML_LOG_ERROR("%s: retained tensor '%s' [%s] overlaps leaf '%s' in buffer %d\n",
                        __func__, node->name, ggml_op_desc(node),
                        graph->leafs[j]->name, kept->buffer_id);
                clash = true;
            }
        }
        if (clash) {
            return false;
        }
    }

    // INFO, not DEBUG: these are the production-path evidence lines. They are
    // emitted by the allocator itself during reserve, so they need no eval
    // callback, no extra synchronize and no change to scheduler timing.
    //
    // The line is per reserve instance, and it says which instance: `entry` is
    // "reserve" for ggml_gallocr_reserve_n and "size" for ggml_gallocr_reserve_n_size
    // (a measurement call that allocates nothing, so it must never be read as an
    // executed-plan guarantee). Nothing here identifies the graph the decode submits:
    // that relation has no source in this path, so a clean line is not evidence for
    // it (see ROUTER_SCRATCH_CPU_FIXTURE.md §9).
    // `rows` is the common ne[1] of the retained projections, "mixed" when they
    // disagree, which attributes the line to a graph shape without relying on
    // the shape being reserved exactly once. The zero-retained case is a
    // separate "enabled" line with no rows field.
    if (n_unverifiable != 0) {
        // Some allocation the retained extent could collide with is outside this
        // plan (it owns its data, or carries no recorded size), so no overlap was
        // ruled out there. Report UNPROVEN and refuse: the retention contract
        // holds only for allocations the plan actually describes.
        GGML_LOG_ERROR("%s: retention audit unproven (%s): %d in-domain allocation(s) carry no "
                "recorded size; disjointness is unproven\n",
                __func__, entry, n_unverifiable);
        return false;
    }

    if (n_retained == 0) {
        // Explicit zero line: proves the audit ran and vouched for nothing.
        // Without it, silence is ambiguous between "no retained allocations"
        // and "audit not enabled / reserve not reached".
        GGML_LOG_INFO("%s: retention audit enabled (%s): 0 retained allocations, nodes=%d, graph_uid=%" PRIu64 ", plan_id=%" PRId64 ", fp=%016" PRIx64 "\n",
                __func__, entry, graph->n_nodes,
                galloc->plan_graph_uid, galloc->plan_id, galloc->plan_fp);
    } else {
        char rows_buf[32];
        if (n1_mixed) {
            snprintf(rows_buf, sizeof(rows_buf), "mixed");
        } else {
            snprintf(rows_buf, sizeof(rows_buf), "%d", n1_rows);
        }
        GGML_LOG_INFO("%s: retention audit clean (%s): %d retained allocations, %zu bytes, rows=%s, nodes=%d, graph_uid=%" PRIu64 ", plan_id=%" PRId64 ", fp=%016" PRIx64 "\n",
                __func__, entry, n_retained, retained_bytes, rows_buf, graph->n_nodes,
                galloc->plan_graph_uid, galloc->plan_id, galloc->plan_fp);
    }

    // The plan passed the audit with retained allocations: this is the identity an
    // owner may compare against. Written last, so a refusal above leaves it cleared.
    //
    // Only a plan that actually carries retention is recorded. A plan with zero
    // retained allocations has nothing for this audit to vouch for, so claiming it
    // would let a 1-row or knob-off reserve be reported as an audited retained plan;
    // those passes leave the record cleared and any owner check fails closed.
    galloc->plan_retained = n_retained;

    if (n_retained > 0) {
        galloc->audit_clean_plan_id   = galloc->plan_id;
        galloc->audit_clean_graph_uid = galloc->plan_graph_uid;
        galloc->audit_clean_fp        = galloc->plan_fp;
        galloc->audit_clean_n_nodes   = galloc->plan_n_nodes;
        galloc->audit_clean_n_leafs   = galloc->plan_n_leafs;
    }
    return true;
}

// free the extra space at the end if the new tensor is smaller
static void ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, struct ggml_tensor * node, struct ggml_tensor * parent) {
    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
    struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);

    size_t parent_size = ggml_backend_buft_get_alloc_size(galloc->bufts[p_hn->buffer_id], parent);
    size_t node_size = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node);

    GGML_ASSERT(parent_size >= node_size);

    // note: we want after the freeing the chunks to continue to be aligned
    struct ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->buffer_id];
    parent_size = aligned_offset(NULL, parent_size, p_alloc->alignment);
    node_size = aligned_offset(NULL, node_size, p_alloc->alignment);

    if (parent_size > node_size) {
        struct buffer_address p_addr = p_hn->addr;
        p_addr.offset += node_size;
        size_t extra_size = parent_size - node_size;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_size, parent->name, node->name);
        ggml_dyn_tallocr_free_bytes(p_alloc, p_addr, extra_size);
    }
}

static void ggml_gallocr_allocate_node(ggml_gallocr_t galloc, struct ggml_tensor * node, int buffer_id) {
    GGML_ASSERT(buffer_id >= 0);
    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);

    if (!ggml_gallocr_is_allocated(galloc, node) && !ggml_impl_is_view(node)) {
        hn->allocated = true;
        assert(hn->addr.offset == 0);

        // try to reuse a parent's buffer (inplace)
        if (ggml_op_can_inplace(node->op)) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                struct ggml_tensor * parent = node->src[i];
                if (parent == NULL) {
                    continue;
                }

                // if the node's data is external, then we cannot re-use it
                if (!ggml_gallocr_is_own(galloc, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, node->name, parent->data);
                    continue;
                }

                // Retained tensors, including their views, cannot be reused in-place.
                if (parent->flags & (GGML_TENSOR_FLAG_OUTPUT | GGML_TENSOR_FLAG_NO_REUSE) ||
                    (parent->view_src != NULL && parent->view_src->flags & (GGML_TENSOR_FLAG_OUTPUT | GGML_TENSOR_FLAG_NO_REUSE))) {
                    AT_PRINTF("not reusing retained parent %s for %s\n", parent->name, node->name);
                    continue;
                }

                if (!ggml_are_same_layout(node, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, node->name);
                    continue;
                }

                struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
                if (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    if (ggml_impl_is_view(parent)) {
                        struct ggml_tensor * view_src = parent->view_src;
                        struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        if (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, node->name);
                            assert(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.offset == p_hn->addr.offset);
                            hn->buffer_id = p_hn->buffer_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, node, view_src);
                            return;
                        }
                    } else {
                        AT_PRINTF("reusing parent %s for %s\n", parent->name, node->name);
                        hn->buffer_id = p_hn->buffer_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, node, parent);
                        return;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
        ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
        size_t size = ggml_backend_buft_get_alloc_size(buft, node);
        hn->buffer_id = buffer_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, size, node);
    }
}

static void ggml_gallocr_free_node(ggml_gallocr_t galloc, struct ggml_tensor * node) {
    // Graph outputs and explicitly retained allocations remain live throughout the graph.
    if (node->flags & (GGML_TENSOR_FLAG_OUTPUT | GGML_TENSOR_FLAG_NO_REUSE)) {
        AT_PRINTF("not freeing retained tensor %s\n", node->name);
        return;
    }

    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
    int buffer_id = hn->buffer_id;
    struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
    ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
    size_t size = ggml_backend_buft_get_alloc_size(buft, node);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, node->name, hn->addr.chunk, hn->addr.offset, size, alloc->chunks[hn->addr.chunk]->n_free_blocks);
#ifdef GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, node);
#endif

    ggml_dyn_tallocr_free_bytes(alloc, hn->addr, size);
    hn->allocated = false;
}

static int get_node_buffer_id(const int * node_buffer_ids, int i) {
    return node_buffer_ids ? node_buffer_ids[i] : 0;
}

static void ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
    // clear hash tables
    ggml_hash_set_reset(&galloc->hash_set);
    memset(galloc->hash_values, 0, sizeof(struct hash_node) * galloc->hash_set.size);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        ggml_gallocr_allocate_node(galloc, leaf, get_node_buffer_id(leaf_buffer_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        if (ggml_impl_is_view(node) && node->op != GGML_OP_NONE) {
            struct ggml_tensor * view_src = node->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        // A retained node needs its own storage for the whole graph. Allocating it
        // here, before any temporary is placed, is what gives it that: a temporary
        // allocated first would own the address, and the retained node could only
        // take an address a temporary has already been freed from -- a legal plan by
        // the dyn_tallocr's rules and an overlap by the retention contract. Views are
        // excluded: they do not own storage (allocate_node returns early for them),
        // and their owner is retained through the NO_REUSE check on reuse below.
        const bool retained_owner = (node->flags & GGML_TENSOR_FLAG_NO_REUSE) != 0 && !ggml_impl_is_view(node);
        if ((node->flags & GGML_TENSOR_FLAG_INPUT) || retained_owner) {
            ggml_gallocr_allocate_node(galloc, graph->nodes[i], get_node_buffer_id(node_buffer_ids, i));
        }

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs, and retained sources for the same reason
            const bool src_retained = (src->flags & GGML_TENSOR_FLAG_NO_REUSE) != 0 && !ggml_impl_is_view(src);
            if ((src->flags & GGML_TENSOR_FLAG_INPUT) || src_retained) {
                ggml_gallocr_allocate_node(galloc, src, get_node_buffer_id(node_buffer_ids, i));
            }
        }
    }

    // allocate tensors
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int buffer_id = get_node_buffer_id(node_buffer_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            ggml_gallocr_allocate_node(galloc, parent, buffer_id);
        }

        // allocate node
        ggml_gallocr_allocate_node(galloc, node, buffer_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(node), node->name);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            AT_PRINTF("%s", parent->name);
            if (j < GGML_MAX_SRC - 1 && node->src[j + 1] != NULL) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            if (p_hn->n_children == 0 && p_hn->n_views == 0) {
                if (ggml_impl_is_view(parent)) {
                    // An output view must retain its source after its last consumer.
                    if (parent->flags & GGML_TENSOR_FLAG_OUTPUT) {
                        continue;
                    }
                    struct ggml_tensor * view_src = parent->view_src;
                    struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    if (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_node(galloc, view_src);
                    }
                }
                else if (p_hn->allocated) {
                    ggml_gallocr_free_node(galloc, parent);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

static bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids, bool no_alloc) {
    size_t min_hash_size = graph->n_nodes + graph->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_size += min_hash_size / 4;

    // initialize hash table
    if (galloc->hash_set.size < min_hash_size) {
        ggml_hash_set_free(&galloc->hash_set);
        galloc->hash_set = ggml_hash_set_new(min_hash_size);
        GGML_ASSERT(galloc->hash_set.keys != NULL);

        free(galloc->hash_values);
        galloc->hash_values = malloc(sizeof(struct hash_node) * galloc->hash_set.size);
        GGML_ASSERT(galloc->hash_values != NULL);
    }

    // reset allocators
    for (int i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // A new planning pass: the plan this instance holds is about to be replaced,
    // so its identity must stop matching anything already handed out. Bumped
    // before the plan is built, so a refused plan leaves the new id behind and
    // an owner that saw the old one sees a mismatch instead of a false match.
    galloc->plan_id += 1;
    galloc->plan_entry_size = no_alloc;
    // until the audit of this pass reports, nothing is known about retention
    galloc->plan_retained = -1;

    // The record of the last accepted plan is kept, not cleared: a pass that is
    // refused or skips the audit must leave the two ids apart on its own, which it
    // does because plan_id has just advanced. Clearing here would instead destroy
    // the only record of the plan that is still installed in the buffers, so a
    // caller could no longer tell whether the plan it submits was ever audited.

    // allocate in hash table
    ggml_gallocr_alloc_graph_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids);

    // set the node_allocs from the hash table
    if (galloc->n_nodes < graph->n_nodes) {
        free(galloc->node_allocs);
        galloc->node_allocs = calloc(graph->n_nodes, sizeof(struct node_alloc));
        GGML_ASSERT(galloc->node_allocs != NULL);
    }
    galloc->n_nodes = graph->n_nodes;
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];
        node_alloc->dst.no_reuse = (node->flags & GGML_TENSOR_FLAG_NO_REUSE) != 0;
        if (node->view_src || node->data) {
            node_alloc->dst.buffer_id = -1;
            node_alloc->dst.addr = GGML_BUFFER_ADDRESS_INVALID;
            node_alloc->dst.size_max = 0;
        } else {
            struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
            node_alloc->dst.buffer_id = hn->buffer_id;
            node_alloc->dst.addr = hn->addr;
            // plan validity is judged against the unaligned logical size, as
            // ggml_gallocr_node_needs_realloc() compares an unaligned node_size:
            // feeding it an aligned size would absorb a real growth of up to
            // alignment-1 bytes and skip the re-reserve the layout needs.
            node_alloc->dst.size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node);
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            node_alloc->src[j].no_reuse = src && (src->flags & GGML_TENSOR_FLAG_NO_REUSE) != 0;
            if (!src || src->view_src || src->data) {
                node_alloc->src[j].buffer_id = -1;
                node_alloc->src[j].addr = GGML_BUFFER_ADDRESS_INVALID;
                node_alloc->src[j].size_max = 0;
            } else {
                struct hash_node * hn = ggml_gallocr_hash_get(galloc, src);
                node_alloc->src[j].buffer_id = hn->buffer_id;
                node_alloc->src[j].addr = hn->addr;
                node_alloc->src[j].size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], src);
            }
        }
    }
    if (galloc->n_leafs < graph->n_leafs) {
        free(galloc->leaf_allocs);
        galloc->leaf_allocs = calloc(graph->n_leafs, sizeof(galloc->leaf_allocs[0]));
        GGML_ASSERT(galloc->leaf_allocs != NULL);
    }
    galloc->n_leafs = graph->n_leafs;
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        struct hash_node * hn = ggml_gallocr_hash_get(galloc, leaf);
        if (leaf->view_src || leaf->data) {
            galloc->leaf_allocs[i].leaf.buffer_id = -1;
            galloc->leaf_allocs[i].leaf.addr = GGML_BUFFER_ADDRESS_INVALID;
            galloc->leaf_allocs[i].leaf.size_max = 0;
        } else {
            galloc->leaf_allocs[i].leaf.buffer_id = hn->buffer_id;
            galloc->leaf_allocs[i].leaf.addr = hn->addr;
            galloc->leaf_allocs[i].leaf.size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], leaf);
        }
    }

    // refuse a plan that violates the retention contract before it is materialized
    if (!ggml_gallocr_check_no_reuse_plan(galloc, graph, no_alloc ? "size" : "reserve")) {
        return false;
    }

    // reallocate buffers if needed
    for (int i = 0; i < galloc->n_buffers; i++) {
        // if the buffer type is used multiple times, we reuse the same buffer
        for (int j = 0; j < i; j++) {
            if (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                break;
            }
        }

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        bool realloc = galloc->buffers[i] == NULL;
        size_t new_size = 0;
        for (int c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_size = galloc->buffers[i] ? ggml_vbuffer_chunk_size(galloc->buffers[i], c) : 0;
            size_t new_chunk_size = ggml_dyn_tallocr_max_size(galloc->buf_tallocs[i], c);
            new_size += new_chunk_size;
            if (new_chunk_size > cur_chunk_size) {
                realloc = true;
            }
        }
        if (realloc) {
#ifndef NDEBUG
            {
                size_t cur_size = galloc->buffers[i] ? ggml_vbuffer_size(galloc->buffers[i]) : 0;
                if (cur_size > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
                }
            }
#endif
            ggml_vbuffer_free(galloc->buffers[i]);
            if (no_alloc) {
                galloc->buffers[i] = NULL;
            } else {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                if (galloc->buffers[i] == NULL) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_size);
                    return false;
                }
            }
        }
    }

    return true;
}

void ggml_gallocr_reserve_n_size(
        ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ true));
    for (int i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        for (int c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_size;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
    return ggml_gallocr_reserve_n_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph *graph) {
    return ggml_gallocr_reserve_n(galloc, graph, NULL, NULL);
}

static void ggml_gallocr_init_tensor(ggml_gallocr_t galloc, struct ggml_tensor * tensor, struct tensor_alloc * tensor_alloc) {
    int buffer_id = tensor_alloc->buffer_id;
    assert(tensor->data || tensor->view_src || ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);

    if (tensor->view_src != NULL) {
        if (tensor->buffer == NULL) {
            assert(tensor_alloc->addr.offset == SIZE_MAX);
            if (tensor->view_src->buffer == NULL) {
                // this tensor was allocated without ggml-backend
                return;
            }
            ggml_backend_view_init(tensor);
        }
    } else {
        if (tensor->data == NULL) {
            assert(tensor_alloc->addr.offset != SIZE_MAX);
            assert(ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[buffer_id], tensor, tensor_alloc->addr);
        } else {
            if (tensor->buffer == NULL) {
                // this tensor was allocated without ggml-backend
                return;
            }
        }
    }
}

static bool ggml_gallocr_node_needs_realloc(ggml_gallocr_t galloc, struct ggml_tensor * node, struct tensor_alloc * talloc) {
    if (talloc->no_reuse != ((node->flags & GGML_TENSOR_FLAG_NO_REUSE) != 0)) {
        return false;
    }
    size_t node_size = 0;
    if (!node->data && !node->view_src) {
        // If we previously had data but don't now then reallocate
        if (talloc->buffer_id < 0) {
            return false;
        }
        node_size = ggml_backend_buft_get_alloc_size(galloc->bufts[talloc->buffer_id], node);
    }
    return talloc->size_max >= node_size;
}

static bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (galloc->n_nodes != graph->n_nodes) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#endif
        return true;
    }

    if (galloc->n_leafs != graph->n_leafs) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#endif
        return true;
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];

        if (!ggml_gallocr_node_needs_realloc(galloc, node, &node_alloc->dst)) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, node->name);
#endif
            return true;
        }

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            if (!ggml_gallocr_node_needs_realloc(galloc, src, &node_alloc->src[j])) {
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, node->name);
#endif
                return true;
            }
        }
    }

    return false;
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (ggml_gallocr_needs_realloc(galloc, graph)) {
        if (galloc->n_buffers == 1) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#endif
            if (!ggml_gallocr_reserve(galloc, graph)) {
                return false;
            }
        } else {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#endif
            return false;
        }
    }

    // reset buffers
    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->buffers[i] != NULL) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        struct leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
    }
    // nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            ggml_gallocr_init_tensor(galloc, src, &node_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, node, &node_alloc->dst);
    }

    return true;
}

void ggml_gallocr_plan_identity(ggml_gallocr_t galloc, struct ggml_gallocr_plan_identity * out) {
    GGML_ASSERT(galloc != NULL);
    GGML_ASSERT(out != NULL);

    out->plan_id    = galloc->plan_id;
    out->graph_uid  = galloc->plan_graph_uid;
    out->fp         = galloc->plan_fp;
    out->n_nodes    = galloc->plan_n_nodes;
    out->n_leafs    = galloc->plan_n_leafs;
    out->entry_size = galloc->plan_entry_size;
    out->retained   = galloc->plan_retained;

    out->audit_clean_plan_id   = galloc->audit_clean_plan_id;
    out->audit_clean_graph_uid = galloc->audit_clean_graph_uid;
    out->audit_clean_fp        = galloc->audit_clean_fp;
    out->audit_clean_n_nodes   = galloc->audit_clean_n_nodes;
    out->audit_clean_n_leafs   = galloc->audit_clean_n_leafs;
}

size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id) {
    GGML_ASSERT(buffer_id >= 0 && buffer_id < galloc->n_buffers);

    if (galloc->buffers[buffer_id] == NULL) {
        return 0;
    }

    for (int i = 0; i < buffer_id; i++) {
        if (galloc->buffers[i] == galloc->buffers[buffer_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            return 0;
        }
    }

    return ggml_vbuffer_size(galloc->buffers[buffer_id]);
}

// utils

static void free_buffers(ggml_backend_buffer_t ** buffers, const size_t * n_buffers) {
    for (size_t i = 0; i < *n_buffers; i++) {
        ggml_backend_buffer_free((*buffers)[i]);
    }
    free(*buffers);
}

static bool alloc_tensor_range(struct ggml_context * ctx,
        struct ggml_tensor * first, struct ggml_tensor * last,
        ggml_backend_buffer_type_t buft, size_t size,
        ggml_backend_buffer_t ** buffers, size_t * n_buffers) {

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), size);
        free_buffers(buffers, n_buffers);
        return false;
    }

    *buffers = realloc(*buffers, sizeof(ggml_backend_buffer_t) * (*n_buffers + 1));
    (*buffers)[(*n_buffers)++] = buffer;

    struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);

    for (struct ggml_tensor * t = first; t != last; t = ggml_get_next_tensor(ctx, t)) {
        enum ggml_status status = GGML_STATUS_SUCCESS;
        if (t->data == NULL) {
            if (t->view_src == NULL) {
                status = ggml_tallocr_alloc(&tallocr, t);
            } else if (t->buffer == NULL) {
                status = ggml_backend_view_init(t);
            }
        } else {
            if (t->view_src != NULL && t->buffer == NULL) {
                // view of a pre-allocated tensor
                status = ggml_backend_view_init(t);
            }
        }
        if (status != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t->name);
            free_buffers(buffers, n_buffers);
            return false;
        }
    }

    return true;
}

static ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft_impl(
        struct ggml_context * ctx, ggml_backend_buffer_type_t buft, size_t * nbytes_total, bool no_alloc) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t max_size = ggml_backend_buft_get_max_size(buft);

    ggml_backend_buffer_t * buffers = NULL;
    size_t n_buffers = 0;
    *nbytes_total = 0;

    size_t cur_buf_size = 0;
    struct ggml_tensor * first = ggml_get_first_tensor(ctx);
    for (struct ggml_tensor * t = first; t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        size_t this_size = 0;
        if (t->data == NULL && t->view_src == NULL) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        if (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            // allocate tensors in the current buffer
            if (!no_alloc && !alloc_tensor_range(ctx, first, t, buft, cur_buf_size, &buffers, &n_buffers)) {
                return NULL;
            }
            first = t;
            *nbytes_total += cur_buf_size;
            cur_buf_size = this_size;
        } else {
            cur_buf_size += this_size;
        }
    }

    // allocate remaining tensors
    if (cur_buf_size > 0) {
        *nbytes_total += cur_buf_size;
        if (!no_alloc && !alloc_tensor_range(ctx, first, NULL, buft, cur_buf_size, &buffers, &n_buffers)) {
            return NULL;
        }
    }

    if (no_alloc) {
        return NULL;
    }

    if (n_buffers == 0) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: all tensors in the context are already allocated\n", __func__);
#endif
        GGML_ASSERT(!buffers);
        return NULL;
    }

    ggml_backend_buffer_t buffer;
    if (n_buffers == 1) {
        buffer = buffers[0];
    } else {
        buffer = ggml_backend_multi_buffer_alloc_buffer(buffers, n_buffers);
    }
    if (buffers) {
        free(buffers); // can be NULL if context is empty or no_alloc
    }
    return buffer;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_size(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    size_t nbytes_total = 0;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft_impl(ctx, buft, &nbytes_total, /*no_alloc=*/ true);
    GGML_ASSERT(!buf);
    return nbytes_total;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    size_t nbytes_total = 0;
    if (ggml_backend_buft_is_meta(buft)) {
        return ggml_backend_meta_alloc_ctx_tensors_from_buft(ctx, buft);
    }
    return ggml_backend_alloc_ctx_tensors_from_buft_impl(ctx, buft, &nbytes_total, /*no_alloc =*/ false);
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend) {
    return ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
}
