#include "ggml-backend-dl.h"
#include "ggml-backend-module.h"
#include "ggml-impl.h"

#include <mutex>
#include <new>

namespace {

struct deferred_node {
    dl_handle * handle;
    deferred_node * next;
};

std::mutex & deferred_handle_mutex() {
    static std::mutex * mutex = new std::mutex();
    return *mutex;
}

deferred_node *& deferred_head() {
    static deferred_node * head = nullptr;
    return head;
}

thread_local bool t_draining_deferred = false;

struct drain_registrar {
    drain_registrar() {
        ggml_backend_module::set_close_drain(&dl_handle_drain_deferred);
    }
} g_drain_registrar;

} // namespace

void dl_handle_defer_close(dl_handle * handle) {
    if (handle == nullptr) {
        return;
    }
    if (!ggml_backend_module::close_deferred() && !t_draining_deferred) {
        // Host code is the only caller: the last reference was dropped outside
        // provider free and outside a registry transaction. Closing here
        // cannot unmap the code that is currently running.
        dl_handle_deleter{}(handle);
        return;
    }
    deferred_node * node = new (std::nothrow) deferred_node{handle, nullptr};
    if (node == nullptr) {
        // No storage for the deferred close. Keep the module mapped rather
        // than dlclosing inside provider code; the OS reclaims it at exit.
        GGML_LOG_ERROR("%s: cannot queue deferred close; module kept mapped\n", __func__);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(deferred_handle_mutex());
        node->next = deferred_head();
        deferred_head() = node;
    }
}

void dl_handle_drain_deferred() {
    if (t_draining_deferred) {
        // A DSO static destructor may load another backend. Do not close a
        // second batch from inside the first batch's dlclose.
        return;
    }
    t_draining_deferred = true;
    // At most two rounds: the first drains the snapshot; the second catches
    // handles queued by static destructors during the first round. Anything
    // queued by the second round stays for the next transaction.
    for (int round = 0; round < 2; ++round) {
        deferred_node * list = nullptr;
        {
            std::lock_guard<std::mutex> lock(deferred_handle_mutex());
            list = deferred_head();
            deferred_head() = nullptr;
        }
        if (list == nullptr) {
            break;
        }
        while (list != nullptr) {
            deferred_node * next = list->next;
            dl_handle_deleter{}(list->handle);
            delete list;
            list = next;
        }
    }
    t_draining_deferred = false;
}

#ifdef _WIN32

dl_handle * dl_load_library(const fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    return handle;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    void * p = (void *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    return p;
}

const char * dl_error() {
    return "";
}

#else

dl_handle * dl_load_library(const fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    return handle;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    return dlsym(handle, name);
}

const char * dl_error() {
    const char *rslt = dlerror();
    return rslt != nullptr ? rslt : "";
}

#endif
