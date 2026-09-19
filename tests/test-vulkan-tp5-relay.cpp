// RELAY hardware execution is fail-closed after an unsafe RADV/Navi21 reset.
#include <cstdio>
#include <cstring>

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--run") == 0) {
            fprintf(stderr, "relay: disabled after an unsafe RADV/Navi21 reset; no GPU work was submitted\n");
            return 1;
        }
    }

    fprintf(stderr, "test-vulkan-tp5-relay is disabled after an unsafe RADV/Navi21 reset; no GPU work is available\n");
    return 0;
}
