#pragma once

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <functional>
#include <stdexcept>
#include <thread>
#include <unistd.h>

// Only the pipe write runs in signal context. Shutdown may take queue locks,
// so invoke the caller's live handler from a normal thread instead.
class server_signal_dispatcher {
public:
    explicit server_signal_dispatcher(std::function<void(int)> & handler) : handler(handler) {
        if (pipe(fds) != 0) {
            throw std::runtime_error("cannot create shutdown signal pipe");
        }
        try {
            for (int fd : fds) {
                if (fcntl(fd, F_SETFD, FD_CLOEXEC) != 0) {
                    throw std::runtime_error("cannot protect shutdown pipe across exec");
                }
            }
            terminating.clear();
            signal_fd = fds[1];
            worker = std::thread([this] {
                unsigned char value;
                for (;;) {
                    ssize_t result = read(fds[0], &value, sizeof(value));
                    if (result < 0 && errno == EINTR) {
                        continue;
                    }
                    if (result != sizeof(value) || value == 0) {
                        return;
                    }
                    this->handler(value);
                }
            });
            struct sigaction action{};
            action.sa_handler = dispatch;
            sigemptyset(&action.sa_mask);
            if (sigaction(SIGINT, &action, &old_interrupt) != 0) {
                throw std::runtime_error("cannot install interrupt dispatcher");
            }
            interrupt_installed = true;
            if (sigaction(SIGTERM, &action, &old_terminate) != 0) {
                throw std::runtime_error("cannot install termination dispatcher");
            }
            terminate_installed = true;
        } catch (...) {
            stop();
            throw;
        }
    }

    server_signal_dispatcher(const server_signal_dispatcher &) = delete;
    server_signal_dispatcher & operator=(const server_signal_dispatcher &) = delete;

    ~server_signal_dispatcher() { stop(); }

    // Join before the handler's captured server state is destroyed. Calling
    // this after normal/API termination must not require an incoming signal.
    void stop() {
        if (terminate_installed) {
            sigaction(SIGTERM, &old_terminate, nullptr);
            terminate_installed = false;
        }
        if (interrupt_installed) {
            sigaction(SIGINT, &old_interrupt, nullptr);
            interrupt_installed = false;
        }
        if (worker.joinable()) {
            const unsigned char stop = 0;
            ssize_t result;
            do {
                result = write(fds[1], &stop, sizeof(stop));
            } while (result < 0 && errno == EINTR);
            if (result != sizeof(stop)) {
                std::terminate();
            }
            worker.join();
        }
        signal_fd = -1;
        for (int & fd : fds) {
            if (fd >= 0) {
                close(fd);
                fd = -1;
            }
        }
    }

private:
    static void dispatch(int signal) {
        if (terminating.test_and_set()) {
            // Keep the second-signal escape hatch even if shutdown blocks.
            std::_Exit(1);
        }
        const int saved_errno = errno;
        const unsigned char value = static_cast<unsigned char>(signal);
        ssize_t result;
        do {
            result = write(signal_fd, &value, sizeof(value));
        } while (result < 0 && errno == EINTR);
        errno = saved_errno;
        if (result != sizeof(value)) {
            std::_Exit(1);
        }
    }

    static inline std::atomic_flag terminating = ATOMIC_FLAG_INIT;
    static inline volatile sig_atomic_t signal_fd = -1;
    std::function<void(int)> & handler;
    int fds[2] = {-1, -1};
    std::thread worker;
    struct sigaction old_interrupt{};
    struct sigaction old_terminate{};
    bool interrupt_installed = false;
    bool terminate_installed = false;
};

#endif
