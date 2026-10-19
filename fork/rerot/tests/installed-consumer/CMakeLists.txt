cmake_minimum_required(VERSION 3.16)
project(TQRerotInstalledConsumer LANGUAGES CXX)

find_package(TQRerot CONFIG REQUIRED)

add_executable(tq-rerot-installed-consumer main.cpp)
target_link_libraries(tq-rerot-installed-consumer PRIVATE TQRerot::rerot)
set_target_properties(tq-rerot-installed-consumer PROPERTIES CXX_EXTENSIONS OFF)

enable_testing()
add_test(NAME rerot-installed-consumer COMMAND tq-rerot-installed-consumer)
