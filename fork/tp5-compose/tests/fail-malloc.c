// Test-only malloc/calloc failure interposer. Build as a shared library and
// load it with LD_PRELOAD. The test arms a countdown immediately before the
// call under test and disarms after; every other allocation passes through.
#define _GNU_SOURCE
#include <stddef.h>
#include <stdlib.h>

#if defined(__GLIBC__)
extern void * __libc_malloc(size_t size);
extern void * __libc_calloc(size_t n, size_t size);
#else
#include <dlfcn.h>
#endif

static void * (*real_malloc)(size_t) = NULL;
static void * (*real_calloc)(size_t, size_t) = NULL;
static int    countdown = -1;
static int    armed = 0;

static int should_fail(void) {
    if (!armed) {
        return 0;
    }
    if (countdown == 0) {
        armed = 0;
        return 1;
    }
    if (countdown > 0) {
        --countdown;
    }
    return 0;
}

void * malloc(size_t size) {
    if (real_malloc == NULL) {
#if defined(__GLIBC__)
        real_malloc = __libc_malloc;
#else
        real_malloc = (void * (*)(size_t)) dlsym(RTLD_NEXT, "malloc");
#endif
        if (real_malloc == NULL) {
            return NULL;
        }
    }
    if (should_fail()) {
        return NULL;
    }
    return real_malloc(size);
}

void * calloc(size_t n, size_t size) {
    if (real_calloc == NULL) {
#if defined(__GLIBC__)
        real_calloc = __libc_calloc;
#else
        real_calloc = (void * (*)(size_t, size_t)) dlsym(RTLD_NEXT, "calloc");
#endif
        if (real_calloc == NULL) {
            return NULL;
        }
    }
    if (should_fail()) {
        return NULL;
    }
    return real_calloc(n, size);
}

void tp5_fail_malloc_arm(int n) {
    countdown = n;
    armed = 1;
}

void tp5_fail_malloc_disarm(void) {
    armed = 0;
}
