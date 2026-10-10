#include "server-queue.h"
#include "server-signal.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <poll.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

static int wait_child(pid_t child) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    int status = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        const pid_t result = waitpid(child, &status, WNOHANG);
        if (result == child) {
            return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        }
        if (result < 0) {
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    kill(child, SIGKILL);
    waitpid(child, &status, 0);
    return -1;
}

static bool terminate_with_queue_lock() {
    const pid_t child = fork();
    if (child < 0) {
        return false;
    }
    if (child == 0) {
        server_queue queue;
        const auto main_thread = std::this_thread::get_id();
        std::atomic<bool> regular_thread{false};
        bool signal_returned = false;
        std::function<void(int)> shutdown = [&](int signal) {
            regular_thread = signal == SIGTERM && std::this_thread::get_id() != main_thread;
            queue.terminate();
        };
        server_signal_dispatcher dispatcher(shutdown);
        queue.on_new_task([](server_task &&) { std::_Exit(2); });
        queue.on_update_slots([] {});
        queue.on_sleeping_state([&](bool sleeping) {
            if (sleeping) {
                // Production start_loop invokes this while holding mutex_tasks.
                raise(SIGTERM);
                signal_returned = true;
            }
        });
        queue.start_loop(0);
        dispatcher.stop();
        std::_Exit(signal_returned && regular_thread.load() ? 0 : 2);
    }
    return wait_child(child) == 0;
}

static bool second_signal_escapes_blocked_shutdown() {
    int ready[2];
    if (pipe(ready) != 0) {
        return false;
    }
    const pid_t child = fork();
    if (child < 0) {
        close(ready[0]);
        close(ready[1]);
        return false;
    }
    if (child == 0) {
        close(ready[0]);
        std::function<void(int)> shutdown = [&](int) {
            const char entered = 'E';
            if (write(ready[1], &entered, 1) != 1) {
                std::_Exit(2);
            }
            for (;;) {
                pause();
            }
        };
        server_signal_dispatcher dispatcher(shutdown);
        raise(SIGINT);
        for (;;) {
            pause();
        }
    }
    close(ready[1]);
    struct pollfd event{ready[0], POLLIN, 0};
    const int available = poll(&event, 1, 2000);
    char entered = 0;
    const bool acknowledged = available == 1 && read(ready[0], &entered, 1) == 1 && entered == 'E';
    close(ready[0]);
    kill(child, acknowledged ? SIGTERM : SIGKILL);
    const int result = wait_child(child);
    return acknowledged && result == 1;
}

static bool normal_stop_restores_handler() {
    const pid_t child = fork();
    if (child < 0) {
        return false;
    }
    if (child == 0) {
        struct sigaction ignore{};
        ignore.sa_handler = SIG_IGN;
        sigemptyset(&ignore.sa_mask);
        if (sigaction(SIGTERM, &ignore, nullptr) != 0) {
            std::_Exit(2);
        }
        std::atomic<int> calls{0};
        std::function<void(int)> shutdown = [&](int) { ++calls; };
        {
            server_signal_dispatcher dispatcher(shutdown);
            dispatcher.stop();
            dispatcher.stop();
        }
        // No synthetic shutdown callback, no join waiting for a signal, and
        // the embedding caller's prior disposition must remain in effect.
        raise(SIGTERM);
        std::_Exit(calls.load() == 0 ? 0 : 2);
    }
    return wait_child(child) == 0;
}

static bool early_termination_is_terminal() {
    const pid_t child = fork();
    if (child < 0) {
        return false;
    }
    if (child == 0) {
        server_queue queue;
        queue.on_new_task([](server_task &&) { std::_Exit(2); });
        queue.on_update_slots([] { std::_Exit(2); });
        queue.terminate();
        queue.start_loop();
        std::_Exit(0);
    }
    return wait_child(child) == 0;
}

int main() {
    if (!terminate_with_queue_lock()) {
        std::fputs("SIGTERM under the real queue mutex did not terminate safely\n", stderr);
        return 1;
    }
    if (!second_signal_escapes_blocked_shutdown()) {
        std::fputs("second signal did not escape blocked shutdown\n", stderr);
        return 1;
    }
    if (!normal_stop_restores_handler()) {
        std::fputs("normal termination did not restore signal ownership\n", stderr);
        return 1;
    }
    if (!early_termination_is_terminal()) {
        std::fputs("shutdown before loop start was lost\n", stderr);
        return 1;
    }
    std::puts("Signal shutdown boundary regressions passed");
    return 0;
}
