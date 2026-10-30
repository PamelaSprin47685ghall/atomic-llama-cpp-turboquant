cmake_minimum_required(VERSION 3.16)
project(TQScheduling VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(TQ_SCHEDULING_BUILD_TESTS "Build the standalone scheduling tests." ON)

add_library(tq_scheduling STATIC src/prefill.cpp src/admission.cpp)
add_library(TQScheduling::scheduling ALIAS tq_scheduling)
set_target_properties(tq_scheduling PROPERTIES
    EXPORT_NAME scheduling
    OUTPUT_NAME tq-scheduling
    POSITION_INDEPENDENT_CODE ON)
target_compile_features(tq_scheduling PUBLIC cxx_std_17)
target_include_directories(tq_scheduling PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(TQ_SCHEDULING_BUILD_TESTS)
    enable_testing()
    add_executable(tq_scheduling_tests tests/prefill-test.cpp)
    target_link_libraries(tq_scheduling_tests PRIVATE TQScheduling::scheduling)
    add_test(NAME tq_scheduling.boundaries COMMAND tq_scheduling_tests)
    add_executable(tq_admission_tests tests/admission-test.cpp)
    target_link_libraries(tq_admission_tests PRIVATE TQScheduling::scheduling)
    add_test(NAME tq_scheduling.admission COMMAND tq_admission_tests)
endif()

set(TQ_SCHEDULING_PACKAGE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/TQScheduling")
configure_package_config_file(cmake/TQSchedulingConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/TQSchedulingConfig.cmake"
    INSTALL_DESTINATION "${TQ_SCHEDULING_PACKAGE_DIR}")
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TQSchedulingConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion)
install(TARGETS tq_scheduling EXPORT TQSchedulingTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(EXPORT TQSchedulingTargets
    FILE TQSchedulingTargets.cmake
    NAMESPACE TQScheduling::
    DESTINATION "${TQ_SCHEDULING_PACKAGE_DIR}")
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TQSchedulingConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/TQSchedulingConfigVersion.cmake"
    DESTINATION "${TQ_SCHEDULING_PACKAGE_DIR}")
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/TQScheduling")
