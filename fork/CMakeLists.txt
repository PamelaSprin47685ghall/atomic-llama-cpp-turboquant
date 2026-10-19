cmake_minimum_required(VERSION 3.16)
project(TQFork VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
find_package(llama CONFIG REQUIRED)
add_subdirectory(sampling)
add_subdirectory(compose)
add_subdirectory(corpus)
add_subdirectory(corpus-compose)
add_subdirectory(scheduling)
add_subdirectory(batch-compose)
add_subdirectory(logits-compose)
# This C consumes installed TQRerot, TQRelay, and TQText packages. Build the
# independent B packages with build.py first; do not substitute sibling targets.
add_subdirectory(rerot-compose)
if(BUILD_TESTING)
    add_subdirectory(tests/a-seams)
endif()
