cmake_minimum_required(VERSION 3.16)
project(TQSampling VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

option(TQ_SAMPLING_BUILD_TESTS "Build the sampling tests." ${BUILD_TESTING})
option(TQ_SAMPLING_SCALAR_ONLY "Disable the AVX2 scan." OFF)

add_library(tq_sampling STATIC src/shortlist.cpp)
add_library(TQSampling::sampling ALIAS tq_sampling)
set_target_properties(tq_sampling PROPERTIES
    EXPORT_NAME sampling
    OUTPUT_NAME tq-sampling
    POSITION_INDEPENDENT_CODE ON)
target_compile_features(tq_sampling PUBLIC cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tq_sampling PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
target_include_directories(tq_sampling PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
if(TQ_SAMPLING_SCALAR_ONLY)
    target_compile_definitions(tq_sampling PRIVATE TQ_SAMPLING_SCALAR_ONLY=1)
endif()

if(TQ_SAMPLING_BUILD_TESTS)
    enable_testing()
    add_executable(tq_sampling_tests tests/shortlist-test.cpp)
    target_link_libraries(tq_sampling_tests PRIVATE TQSampling::sampling)
    add_test(NAME tq_sampling.boundaries COMMAND tq_sampling_tests)
endif()

set(TQ_SAMPLING_PACKAGE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/TQSampling")
configure_package_config_file(cmake/TQSamplingConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/TQSamplingConfig.cmake"
    INSTALL_DESTINATION "${TQ_SAMPLING_PACKAGE_DIR}")
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TQSamplingConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion)

install(TARGETS tq_sampling EXPORT TQSamplingTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(EXPORT TQSamplingTargets
    FILE TQSamplingTargets.cmake
    NAMESPACE TQSampling::
    DESTINATION "${TQ_SAMPLING_PACKAGE_DIR}")
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TQSamplingConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/TQSamplingConfigVersion.cmake"
    DESTINATION "${TQ_SAMPLING_PACKAGE_DIR}")
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/TQSampling")
