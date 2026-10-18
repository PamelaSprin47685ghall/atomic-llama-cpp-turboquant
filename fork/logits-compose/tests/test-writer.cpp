#include "tape-writer.h"

#include <csignal>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <type_traits>
#include <sys/resource.h>
#include <sys/wait.h>

namespace fs = std::filesystem;

static_assert(!std::is_copy_constructible<tq_logits::tape_writer>::value, "a tape must have one row counter");
static_assert(!std::is_move_constructible<tq_logits::tape_writer>::value, "a tape must keep its output owner");

void check(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

void rejects(const std::function<void()> & action) {
    bool rejected = false;
    try { action(); } catch (const std::exception &) { rejected = true; }
    check(rejected, "operation should fail");
}

std::vector<unsigned char> read(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void no_temporaries(const fs::path & directory) {
    for (const auto & entry : fs::directory_iterator(directory)) {
        check(entry.path().filename().string().rfind(".tq-logits-", 0) != 0, "temporary file remains");
    }
}

void run(const fs::path & directory) {
    const auto path = directory / "valid.bin";
    const std::vector<std::int32_t> tokens = {1, 2, 3};
    const float first[] = {0.0f, -0.0f, 1.0f, -2.0f};
    const float second[] = {0.5f, 2.0f, -1.0f, 4.0f};
    {
        tq_logits::new_file file(path);
        tq_logits::tape_writer tape(file, 4, 2, tokens, 68);
        check(tape.bytes() == 68, "incorrect tape size");
        tape.append(first);
        tape.append(second);
        check(!fs::exists(path), "output was visible before publication");
        tape.finish();
        check(fs::exists(path), "output not published");
        rejects([&] { tape.finish(); });
    }
    const std::vector<unsigned char> expected = {
        'T','P','5','L','O','G','2',0, 4,0,0,0, 2,0,0,0, 1,0,0,0, 3,0,0,0,
        1,0,0,0, 2,0,0,0, 3,0,0,0,
        0,0,0,0, 0,0,0,128, 0,0,128,63, 0,0,0,192,
        0,0,0,63, 0,0,0,64, 0,0,128,191, 0,0,128,64};
    check(read(path) == expected, "binary tape differs from the little-endian fixture");
    rejects([&] { tq_logits::new_file file(path); });
    check(read(path) == expected, "existing output changed");

    const auto missing = directory / "missing.bin";
    for (unsigned kind = 0; kind < 9; ++kind) {
        rejects([&] {
            tq_logits::new_file file(missing);
            if (kind == 0) { tq_logits::tape_writer tape(file, 4, 2, tokens, 67); }
            else if (kind == 1) { tq_logits::tape_writer tape(file, 0, 2, tokens, 1024); }
            else if (kind == 2) { tq_logits::tape_writer tape(file, 4, 0, tokens, 1024); }
            else if (kind == 3) { tq_logits::tape_writer tape(file, 4, 4, tokens, 1024); }
            else if (kind == 4) { tq_logits::tape_writer tape(file, 4, 1, {-1}, 1024); }
            else if (kind == 5) { tq_logits::tape_writer tape(file, 4, 1, {4}, 1024); }
            else {
                tq_logits::tape_writer tape(file, 4, 2, tokens, 1024);
                if (kind == 6) { tape.append(first); tape.finish(); }
                if (kind == 7) {
                    const float invalid[] = {0, 1, std::numeric_limits<float>::quiet_NaN(), 2};
                    tape.append(invalid);
                }
                if (kind == 8) { tape.append(first); tape.append(second); tape.append(first); }
            }
        });
        check(!fs::exists(missing), "failed tape was published");
        no_temporaries(directory);
    }

    const auto symlink = directory / "link.bin";
    fs::create_symlink(missing, symlink);
    rejects([&] { tq_logits::new_file file(symlink); });
    check(fs::is_symlink(symlink) && !fs::exists(missing), "broken symlink changed");
    fs::remove(symlink);
    fs::create_symlink(path, symlink);
    rejects([&] { tq_logits::new_file file(symlink); });
    check(read(path) == expected, "symlink target changed");

    const auto race = directory / "race.bin";
    {
        tq_logits::new_file one(race);
        tq_logits::new_file two(race);
        one.write("first", 5);
        two.write("second", 6);
        one.publish();
        rejects([&] { two.publish(); });
    }
    check(read(race) == std::vector<unsigned char>({'f','i','r','s','t'}), "publication race overwrote output");

    const auto collision = directory / (".tq-logits-" + std::to_string(::getpid()) + "-0.tmp");
    {
        tq_logits::new_file file(collision);
        file.write("complete", 8);
        check(!fs::exists(collision), "temporary used the final output name");
        file.publish();
    }
    check(read(collision).size() == 8, "temporary-like output name failed");
    fs::remove(collision);

    const pid_t child = ::fork();
    check(child >= 0, "fork failed");
    if (child == 0) {
        std::signal(SIGXFSZ, SIG_IGN);
        const struct rlimit limit = {8, 8};
        if (::setrlimit(RLIMIT_FSIZE, &limit) != 0) { ::_exit(2); }
        try {
            tq_logits::new_file file(missing);
            try { tq_logits::tape_writer tape(file, 4, 2, tokens, 1024); }
            catch (const std::system_error &) {
                rejects([&] { file.write("retry", 5); });
                rejects([&] { file.publish(); });
                throw;
            }
        } catch (const std::system_error &) { ::_exit(0); }
        ::_exit(3);
    }
    int status = 0;
    check(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0, "write failure was not detected");
    check(!fs::exists(missing), "write failure published output");
    no_temporaries(directory);
}

int main() {
    char pattern[] = "tq-logits-writer-XXXXXX";
    const char * name = ::mkdtemp(pattern);
    if (!name) { std::cerr << "cannot create test directory\n"; return 1; }
    const fs::path directory(name);
    int result = 0;
    try { run(directory); std::cout << "writer: all checks passed\n"; }
    catch (const std::exception & error) { std::cerr << error.what() << '\n'; result = 1; }
    fs::remove_all(directory);
    return result;
}
