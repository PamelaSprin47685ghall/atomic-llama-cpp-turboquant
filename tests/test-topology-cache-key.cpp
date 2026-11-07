//
// test-topology-cache-key.cpp
//
// Proves the single graph cache key of src/llama-topology-hash.h: the
// llama_topology_key struct, llama_topology_key_of() and llama_topology_same().
//
// The key exists so that every reuse question is phrased in one identity. Its
// contract has two halves that point in opposite directions:
//
//   invariant to the runtime parameter values: the key never reads n_kv,
//   n_outputs, rs_z or n_tokens. Two graphs built at the same shapes have the
//   same key whatever those parameters were.
//
//   sensitive to the graph as built: those parameters reach the graph as
//   tensor shapes and view offsets, so a graph built at a wider KV cache IS a
//   different graph and the key must say so.
//
// This file tests the key itself, from ggml ops only: no model, no backend, no
// GPU, and it links ggml alone. The half that needs the real binder - a length
// sweep with rebuild_count == 0 - is tests/test-topology-key-sweep.cpp, which
// links llama and needs a model.
//
// Every CHECK below is falsifiable; the comment above each case says what
// would have to be true for it to fail.
//

#include "../src/llama-topology-hash.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    } \
} while (0)

// ---------------------------------------------------------------------------
// graph builder
// ---------------------------------------------------------------------------

// The three length parameters the key must be invariant to, as values used to
// size the graph. `n_kv` sizes the KV cache leaf, `n_out` sizes the out_ids
// leaf, `rs_z` picks the recurrent-state view offset. Two builds with the same
// values are the same graph; a build with a different value is a different
// graph.
struct shape {
    int64_t n_kv   = 64;
    int64_t n_out  = 8;
    int64_t rs_z   = 0;
    int64_t n_embd = 32;
    int64_t n_head = 4;
    int64_t hd     = 8;
};

struct built {
    ggml_context * ctx = nullptr;
    ggml_cgraph  * gf  = nullptr;
};

// The one builder. Every case calls it; there is no second copy of the graph.
//
// `names` gives every tensor a name. Names are explicitly NOT hashed (see the
// header comment), so a name difference must not move the key - that is the
// falsifiable arm in test_key_ignores_names_and_addresses.
//
// `pad` shifts the arena so every tensor address differs between two builds.
// Addresses are explicitly NOT hashed either.
//
// The graph is the minimum that exercises the three length parameters: a
// KV-shaped fixed leaf sized by n_kv, an out_ids-shaped fixed leaf sized by
// n_out, and a recurrent-state view offset by rs_z. It is deliberately small -
// the point is the key, not a decoder.
static ggml_cgraph * build_into(ggml_context * ctx, const shape & s, bool names) {
    auto nm = [&](ggml_tensor * t, const char * name) -> ggml_tensor * {
        if (names) {
            ggml_set_name(t, name);
        }
        return t;
    };

    const int64_t n_embd = s.n_embd;
    const int64_t n_head = s.n_head;
    const int64_t hd     = s.hd;

    // Fixed leaves: storage pre-exists the build. The digest's leaf predicate
    // is `view_src == nullptr && (data != nullptr || buffer != nullptr)`, so
    // mark them with a non-null `data` sentinel. It is never dereferenced:
    // nothing here computes and the digest never reads `data`.
    ggml_tensor * w   = nm(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd), "w.weight");
    ggml_tensor * k   = nm(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, s.n_kv, n_head, 1), "k_cache");
    ggml_tensor * v   = nm(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, s.n_kv, n_head, 1), "v_cache");
    ggml_tensor * rs  = nm(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 4), "rs");
    ggml_tensor * oid = nm(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, s.n_out), "out_ids");
    ggml_tensor * inp = nm(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1), "inp");

    for (ggml_tensor * t : { w, k, v, rs, oid, inp }) {
        t->data = &t;
    }

    // The recurrent-state zero view: a row offset taken from rs_z.
    ggml_tensor * z  = nm(ggml_view_2d(ctx, rs, n_embd, 1, rs->nb[1], (size_t) s.rs_z * rs->nb[1]), "rs_zero");
    ggml_tensor * zs = nm(ggml_scale(ctx, z, 0.0f), "rs_zero_scaled");

    // A read of the whole KV cache, so its width is a live shape in the graph.
    ggml_tensor * k_all = nm(ggml_reshape_2d(ctx, k, hd * n_head, s.n_kv), "k_all");
    ggml_tensor * v_all = nm(ggml_reshape_2d(ctx, v, hd * n_head, s.n_kv), "v_all");

    ggml_tensor * q    = nm(ggml_mul_mat(ctx, w, inp), "q");        // [n_embd, 1]
    ggml_tensor * kq   = nm(ggml_mul_mat(ctx, k_all, q), "kq");     // [n_kv, 1]
    ggml_tensor * sm   = nm(ggml_soft_max(ctx, kq), "sm");          // [n_kv, 1]
    // The value product needs the transposed layout, as the live graph builds
    // it: llama-graph.cpp:3341-3343 does
    // `v = ggml_cont(ctx0, ggml_transpose(ctx0, v))` when the cache is not
    // stored transposed, because ggml_can_mul_mat requires a->ne[0] ==
    // b->ne[0] and the scores carry n_kv there, not hd*n_head. Without the
    // transpose this call aborts (ggml.c:3369).
    ggml_tensor * v_t  = nm(ggml_cont(ctx, ggml_transpose(ctx, v_all)), "v_t"); // [n_kv, hd*n_head]
    ggml_tensor * vt   = nm(ggml_mul_mat(ctx, v_t, sm), "vt");      // [hd*n_head, 1]
    ggml_tensor * sum  = nm(ggml_add(ctx, vt, zs), "sum");
    ggml_tensor * proj = nm(ggml_mul_mat(ctx, w, sum), "proj");
    ggml_tensor * rows = nm(ggml_get_rows(ctx, proj, oid), "rows");
    ggml_tensor * out  = nm(ggml_scale(ctx, rows, 0.5f), "out");

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(gf, out);
    return gf;
}

// A build owns its arena, so the caller frees the context and the graph dies
// with it.
static built build_graph(const shape & s, bool names, size_t pad) {
    built b;

    ggml_init_params ip = {
        /* .mem_size   = */ pad + ggml_tensor_overhead() * 32 +
                            ggml_graph_overhead_custom(64, false) + 4096,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    b.ctx = ggml_init(ip);
    if (b.ctx == nullptr) {
        std::fprintf(stderr, "ggml_init failed\n");
        return b;
    }
    b.gf = build_into(b.ctx, s, names);
    return b;
}

static void free_built(built & b) {
    if (b.ctx != nullptr) {
        ggml_free(b.ctx);
        b.ctx = nullptr;
    }
    b.gf = nullptr;
}

// ---------------------------------------------------------------------------
// cases
// ---------------------------------------------------------------------------

// The structural half: the key is a function of the graph, not of the arena
// and not of the labels.
//
// This is the honest pure-ggml statement of "the key reads no parameter". You
// cannot vary n_kv as a *parameter* at fixed graph shapes in pure ggml - n_kv
// sizes the KV leaf, so changing it changes the graph, which is the sensitivity
// half below. What you can prove here is the other side: the two things the
// header says are NOT hashed - tensor names and tensor addresses - really do
// not move the key.
//
// The end-to-end length sweep, against the real binder, is
// tests/test-topology-key-sweep.cpp. This file proves the structural half only.
//
// Fails if `name` or `data`/`buffer` (address) ever reaches the digest.
static void test_key_ignores_names_and_addresses() {
    const shape s;

    // Reference: unnamed, unpadded arena.
    built ref = build_graph(s, /*names=*/false, /*pad=*/0);
    CHECK(ref.ctx != nullptr && ref.gf != nullptr);
    if (ref.ctx == nullptr || ref.gf == nullptr) {
        free_built(ref);
        return;
    }
    const llama_topology_key key = llama_topology_key_of(ref.gf);

    // Same graph, named, in a padded arena: every tensor address differs and
    // every tensor carries a name. Fails if either reaches the digest.
    {
        built named = build_graph(s, /*names=*/true, /*pad=*/4096);
        CHECK(named.ctx != nullptr && named.gf != nullptr);
        if (named.ctx != nullptr && named.gf != nullptr) {
            // The addresses really are different, otherwise this proves
            // nothing about address independence.
            CHECK(ggml_graph_node(named.gf, 0) != ggml_graph_node(ref.gf, 0));
            // The two builds must differ in their NAMES, otherwise the
            // equality below proves nothing about name independence. The
            // premise is a strict inequality, not "the other build has none":
            // some graph nodes carry a name even when this fixture names
            // nothing (measured: ref_named != 0 on this tree), so requiring
            // zero would assert a property of ggml, not of the key.
            // Every graph node carries a name in BOTH builds: ggml names its
            // own results, so counting non-empty names cannot separate the two
            // (measured: 14/14 in each). What separates them is the CONTENT:
            // the named build carries this fixture's labels, the other carries
            // ggml's. The premise of the equality below is therefore "the name
            // arrays differ", counted position by position -- the two graphs
            // are built by the same builder at the same shapes, so the node
            // order is the same and the comparison is meaningful.
            int name_diffs = 0;
            const int n_named = ggml_graph_n_nodes(named.gf);
            const int n_ref   = ggml_graph_n_nodes(ref.gf);
            CHECK(n_named == n_ref);
            for (int i = 0; i < n_named && i < n_ref; ++i) {
                if (std::strcmp(ggml_graph_node(named.gf, i)->name,
                                ggml_graph_node(ref.gf, i)->name) != 0) {
                    ++name_diffs;
                }
            }
            std::fprintf(stderr, "TOPOLOGY_CACHE_KEY names differing=%d of %d\n",
                    name_diffs, n_ref);
            CHECK(name_diffs > 0);
            CHECK(llama_topology_same(key, llama_topology_key_of(named.gf)));
        }
        free_built(named);
    }

    // Same graph, unnamed, a different padded arena. Fails if the key depends
    // on an address alone.
    {
        built moved = build_graph(s, /*names=*/false, /*pad=*/8192);
        CHECK(moved.ctx != nullptr && moved.gf != nullptr);
        if (moved.ctx != nullptr && moved.gf != nullptr) {
            CHECK(ggml_graph_node(moved.gf, 0) != ggml_graph_node(ref.gf, 0));
            CHECK(llama_topology_same(key, llama_topology_key_of(moved.gf)));
        }
        free_built(moved);
    }

    // The node count. The canonical set is a SUPERSET of the graph's node list:
    // the walk also collects the non-fixed sources of graph nodes (the
    // transpose view under its ggml_cont, for instance), which are hashed as
    // members but are not graph nodes. So the key's count is at least the
    // graph's, and the difference is exactly those sources.
    {
        const uint32_t graph_nodes = (uint32_t) ggml_graph_n_nodes(ref.gf);
        CHECK(key.n_nodes >= graph_nodes);
        CHECK(key.n_nodes > 0);
        std::fprintf(stderr,
                "TOPOLOGY_CACHE_KEY nodes key=%u graph=%u extra_sources=%u\n",
                key.n_nodes, graph_nodes, key.n_nodes - graph_nodes);
    }

    free_built(ref);
}

// The sensitivity half: each of the three length parameters reaches the key
// only through the graph it builds. For each one, two graphs that differ in
// that parameter and in nothing else must have different keys, and the same
// node count - so the difference is a shape, not a different graph size.
//
// Fails if the key ignores a shape the parameter produces, or if the
// difference is secretly a different number of nodes (which would make the
// case prove something other than what it claims).
static void test_length_parameters_reach_the_key_as_shapes() {
    const shape base;

    struct variant {
        const char * what;
        shape        s;
    };
    shape wider_kv  = base; wider_kv.n_kv  = 1000;
    shape wider_out = base; wider_out.n_out = 32;
    shape shifted_z = base; shifted_z.rs_z  = 1;

    const variant variants[] = {
        { "n_kv",  wider_kv  },
        { "n_out", wider_out },
        { "rs_z",  shifted_z },
    };

    built ref = build_graph(base, /*names=*/false, /*pad=*/0);
    CHECK(ref.ctx != nullptr && ref.gf != nullptr);
    if (ref.ctx == nullptr || ref.gf == nullptr) {
        free_built(ref);
        return;
    }
    const llama_topology_key kref = llama_topology_key_of(ref.gf);

    for (const variant & v : variants) {
        built other = build_graph(v.s, /*names=*/false, /*pad=*/0);
        CHECK(other.ctx != nullptr && other.gf != nullptr);
        if (other.ctx != nullptr && other.gf != nullptr) {
            const llama_topology_key ko = llama_topology_key_of(other.gf);
            // The parameter moved the key. Fails if the key ignores the shape.
            CHECK(!llama_topology_same(kref, ko));
            // ... and it moved it as a shape, not as a different graph size.
            // Fails if the two graphs differ in their node set.
            CHECK(kref.n_nodes == ko.n_nodes);
            std::fprintf(stderr, "  %s: key %016" PRIx64 " -> %016" PRIx64 ", n_nodes %u\n",
                         v.what, kref.hash, ko.hash, ko.n_nodes);
        }
        free_built(other);
    }

    free_built(ref);
}

// The two halves in one place, which is what the header's contract says: the
// same graph built twice with different names and different addresses is the
// same key; the same graph built at a different n_kv is not.
//
// Fails if either half is dropped - a key that hashed names would fail the
// first, a key that ignored shapes would fail the second.
static void test_contract_both_halves() {
    const shape s;
    shape wider = s;
    wider.n_kv = 1000;

    built a = build_graph(s,     /*names=*/false, /*pad=*/0);
    built b = build_graph(s,     /*names=*/true,  /*pad=*/2048);
    built c = build_graph(wider, /*names=*/true,  /*pad=*/2048);
    CHECK(a.gf != nullptr && b.gf != nullptr && c.gf != nullptr);
    if (a.gf != nullptr && b.gf != nullptr && c.gf != nullptr) {
        CHECK(llama_topology_same(llama_topology_key_of(a.gf), llama_topology_key_of(b.gf)));
        CHECK(!llama_topology_same(llama_topology_key_of(a.gf), llama_topology_key_of(c.gf)));
    }
    free_built(a);
    free_built(b);
    free_built(c);
}

// The null and empty graphs. Fails if a null graph produced a key that could
// be mistaken for a real one, or if the empty graph's digest moved.
static void test_null_and_empty() {
    const llama_topology_key null_key = llama_topology_key_of(nullptr);
    CHECK(null_key.hash == LLAMA_TOPOLOGY_FNV_OFFSET);
    CHECK(null_key.n_nodes == 0);

    ggml_init_params ip = {
        /* .mem_size   = */ ggml_tensor_overhead() * 2 + ggml_graph_overhead_custom(8, false) + 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(ip);
    CHECK(ctx != nullptr);
    if (ctx != nullptr) {
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8, false);
        const llama_topology_key empty = llama_topology_key_of(gf);
        CHECK(empty.hash == LLAMA_TOPOLOGY_FNV_OFFSET);
        CHECK(empty.n_nodes == 0);
        CHECK(ggml_graph_n_nodes(gf) == 0);
        ggml_free(ctx);
    }
}

// The wrapper is the same function, not a second one. Fails if
// llama_topology_hash() and llama_topology_key_of().hash ever disagree.
static void test_wrapper_is_the_same_function() {
    built b = build_graph(shape{}, /*names=*/false, /*pad=*/0);
    CHECK(b.ctx != nullptr && b.gf != nullptr);
    if (b.ctx != nullptr && b.gf != nullptr) {
        CHECK(llama_topology_hash(b.gf) == llama_topology_key_of(b.gf).hash);
    }
    free_built(b);

    CHECK(llama_topology_hash(nullptr) == llama_topology_key_of(nullptr).hash);
}

// llama_topology_same() compares the node count as well as the hash. Fails if
// the node-count guard were dropped: two keys with equal hashes but different
// counts must not compare equal.
static void test_same_compares_the_node_count() {
    llama_topology_key a;
    a.hash    = 0x0123456789abcdefULL;
    a.n_nodes = 10;

    llama_topology_key b = a;
    CHECK(llama_topology_same(a, b));

    b.n_nodes = 11;
    CHECK(!llama_topology_same(a, b));

    b = a;
    b.hash = 0x0123456789abcdeeULL;
    CHECK(!llama_topology_same(a, b));
}

int main() {
    test_key_ignores_names_and_addresses();
    test_length_parameters_reach_the_key_as_shapes();
    test_contract_both_halves();
    test_null_and_empty();
    test_wrapper_is_the_same_function();
    test_same_compares_the_node_count();

    if (g_failures != 0) {
        std::fprintf(stderr, "FAIL %d/%d topology-cache-key checks\n", g_failures, g_checks);
        return 1;
    }
    std::printf("PASS %d topology-cache-key checks\n", g_checks);
    return 0;
}
