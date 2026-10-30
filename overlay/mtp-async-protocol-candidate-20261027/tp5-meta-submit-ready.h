#pragma once
#include <ggml-backend.h>
#ifdef __cplusplus
extern "C" {
#endif
// Scoped to the calling thread. Caller owns userdata until it clears the hook.
// The callback runs after private conversion/prepare and before native submit.
GGML_BACKEND_API bool ggml_backend_meta_set_submit_ready_hook(bool (*hook)(void *), void * userdata);
// Runs only after a successful, sealed native submit and after the provider lock is released.
GGML_BACKEND_API bool ggml_backend_meta_set_submitted_hook(bool (*hook)(void *), void * userdata);
#ifdef __cplusplus
}
#endif
