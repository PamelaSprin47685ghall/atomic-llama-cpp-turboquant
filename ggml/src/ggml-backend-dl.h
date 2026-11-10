#pragma once

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#else
#    include <dlfcn.h>
#    include <unistd.h>
#endif
#include <filesystem>
#include <memory>

namespace fs = std::filesystem;

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

struct dl_handle_deleter {
    void operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

#else

using dl_handle = void;

struct dl_handle_deleter {
    void operator()(void * handle) {
        dlclose(handle);
    }
};

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

// A shared handle must never close inline while provider free code or a
// registry transaction is still on the stack. The deleter queues the raw
// handle instead; dl_handle_drain_deferred() closes queued handles from host
// code after the transaction has released its locks.
void dl_handle_defer_close(dl_handle * handle);
void dl_handle_drain_deferred();

struct dl_handle_deferred_deleter {
    void operator()(dl_handle * handle) const {
        dl_handle_defer_close(handle);
    }
};

class dl_handle_shared : public std::shared_ptr<dl_handle> {
public:
    dl_handle_shared() = default;
    dl_handle_shared(const dl_handle_shared &) = default;
    dl_handle_shared(dl_handle_shared &&) noexcept = default;
    dl_handle_shared & operator=(const dl_handle_shared &) = default;
    dl_handle_shared & operator=(dl_handle_shared &&) noexcept = default;

    // The only way to take ownership of a raw dl_load_library() result. The
    // deferred deleter is installed here, so no caller can accidentally use
    // the shared_ptr default deleter on an OS handle.
    static dl_handle_shared adopt(dl_handle * raw) {
        dl_handle_shared out;
        static_cast<std::shared_ptr<dl_handle> &>(out) =
            std::shared_ptr<dl_handle>(raw, dl_handle_deferred_deleter{});
        return out;
    }
};

dl_handle * dl_load_library(const fs::path & path);
void * dl_get_sym(dl_handle * handle, const char * name);
const char * dl_error();

