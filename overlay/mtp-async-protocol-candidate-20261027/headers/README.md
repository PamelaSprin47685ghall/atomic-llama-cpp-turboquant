# Header closure for the delivered overlay runtime (runtime.cpp)

## What the recorded build compiled against

`compile.argv.json` for this lane carries exactly three `-I` roots:

1. `.../mtp-prompt-hc-consolidated-20261024/headers/src` — llama *private* headers (llama-*.h)
2. `.../mtp-prompt-hc-consolidated-20261024/headers/fork/tp5-compose/include` — `tq/**`
3. `.../mtp-prompt-hc-consolidated-20261024/headers/build-tp5-dual50-20261022/hc-router-owner-a-install/include`
   — the ggml + llama *public* headers (`ggml.h`, `ggml-backend.h`, `ggml-relay.h`, `gguf.h`,
   `llama.h`, `ggml-alloc/cpu/cpp/opt.h`) and the TQ installs (`tq/**`)

There is **no `ggml-cuda.h` in root 3**. The delivered `runtime.cpp` used to `#include
<ggml-cuda.h>`; it used nothing from that header (the q8-cache probe goes through
`ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_fusion_count")` and a locally
declared function type, the same pattern as `fork/tests/a-seams/native-q2-mmvq.cpp` and
`fork/tests/a-seams/q2-f32-glu.cpp`, neither of which includes `ggml-cuda.h`). That include is
removed, so the recorded include set is sufficient as recorded. No new `-I` is needed, and the
CUDA/HIP header trees stay out of this translation unit.

## What root 3 is missing (the real closure gap)

Compared with the tree's delivered headers, root 3's copies differ **only** in the candidate's
own additions (verified by `diff`; no unrelated content):

| header | recorded (root 3) sha256 | delivered sha256 | delta |
|---|---|---|---|
| `ggml-backend.h` | `4dc9e1da84bd9b45e2ac82710caefe3971b621bdb1bd9f0c2fef03bb4f31ed21` | `dc619c7367f64dcf3fd683639c0f1183e6c78dc8139f897a0363ab673d05230e` | 1 hunk: `ggml_backend_meta_output_record/wait`, `..._first_child_device/backend`, `..._first_event_child_backend` |
| `llama.h` | `bad66f3437131f8219070587d366111c325695b17caacccdbf90b6e8e56c9290` | `c0aad9efd9eb1167b1240da95b674e7d28997f16e4964dc56fdc5b87c10f3510` | 1 hunk: `llama_tp5_output_copy_record/event_wait`, `llama_get_logits_ith_no_sync` |

`ggml_backend_meta_first_child_device` takes the Meta **backend** and returns the child
**device**; the runtime passes `native.back()` (the backend) exactly like
`ggml_backend_is_meta` / `ggml_backend_meta_simple_backend`, and keeps the Meta device only for
`ggml_backend_event_new`; the per-window event itself is created from the child device (see
`ggml_backend_meta_first_event_child_backend` for the fallback when child 0 cannot create one).
The rank-event accessors of an intermediate revision were retracted (a HIP handle is not a
`ggml_backend_event_t`), so the header no longer includes `ggml-relay.h` and no rank-event
declaration remains. `ggml_backend_meta_first_child_device` / `..._first_child_backend` /
`..._first_event_child_backend` ARE declared here: runtime.cpp resolves them through this header
only and never includes the private `ggml-backend-impl.h`. This copy must match tree
`ggml/include/ggml-backend.h` byte for byte.

The other recorded headers are byte-identical to the tree delivered set (`ggml.h`,
`ggml-relay.h`, `gguf.h`, `ggml-alloc.h`, `ggml-cpu.h`, `ggml-cpp.h`, `ggml-opt.h`), so they
need no patch.

## How to close it (source-accurate, in a build tree only)

In the build tree (never in this read-only snapshot), replace root 3's two headers with the
delivered copies, which are archived here as the patch results:

    base/<header>   --(<header>.patch)-->  <header>

| file | sha256 | role |
|---|---|---|
| `base/ggml-backend.h` | `4dc9e1da…` | recorded root-3 copy (pre-image, archived verbatim) |
| `ggml-backend.patch` | `654a568e…` | pre-image -> delivered (candidate additions only) |
| `ggml-backend.h` | `dc619c73…` | delivered copy, identical to tree `ggml/include/ggml-backend.h` |
| `base/llama.h` | `bad66f34…` | recorded root-3 copy (pre-image, archived verbatim) |
| `llama.patch` | `8694829d…` | pre-image -> delivered (candidate additions only) |
| `llama.h` | `c0aad9ef…` | delivered copy, identical to tree `include/llama.h` |

`ggml_backend_meta_first_child_device` and the `llama_tp5_*` entry points are exported by the
patched objects (`libggml-base`, `libllama`); the header replaces the declaration surface only.
The delivered `llama.h` declarations are backed by `src/llama-context.cpp` + `include/llama.h`
in this tree, and the accessor by both Meta sources, so a build tree needs a `libggml-base` and
`libllama` relink, not a new library.

Alternative (equally source-accurate): add root 3's `include` directory **after** the tree's
`ggml/include` and `include` in the include order, so the tree headers win. That changes search
order only; it is not used by these patches because the lane's recorded argv order is fixed.

## Verify (CPU; runs no build — for Devops)

    cd <overlay dir>
    patch -o /tmp/hb.cpp -p0 headers/base/ggml-backend.h -i headers/ggml-backend.patch \
      && sha256sum /tmp/hb.cpp headers/ggml-backend.h
    patch -o /tmp/hl.cpp -p0 headers/base/llama.h -i headers/llama.patch \
      && sha256sum /tmp/hl.cpp headers/llama.h
    sha256sum headers/ggml-backend.h headers/llama.h   # == <tree>/ggml/include/ggml-backend.h, <tree>/include/llama.h
