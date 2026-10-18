cmake_minimum_required(VERSION 3.16)
project(TQText VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

add_library(tq_text STATIC src/stream.cpp)
add_library(TQText::text ALIAS tq_text)
set_target_properties(tq_text PROPERTIES
    EXPORT_NAME text
    OUTPUT_NAME tq-text
    POSITION_INDEPENDENT_CODE ON)
target_compile_features(tq_text PUBLIC cxx_std_17)
target_include_directories(tq_text PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(BUILD_TESTING)
    add_executable(tq_text_tests tests/stream-test.cpp)
    target_link_libraries(tq_text_tests PRIVATE TQText::text)
    add_test(NAME tq_text.stream COMMAND tq_text_tests)
endif()

set(TQ_TEXT_PACKAGE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/TQText")
configure_package_config_file(cmake/TQTextConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/TQTextConfig.cmake"
    INSTALL_DESTINATION "${TQ_TEXT_PACKAGE_DIR}")
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TQTextConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion)
install(TARGETS tq_text EXPORT TQTextTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(EXPORT TQTextTargets
    FILE TQTextTargets.cmake
    NAMESPACE TQText::
    DESTINATION "${TQ_TEXT_PACKAGE_DIR}")
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TQTextConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/TQTextConfigVersion.cmake"
    DESTINATION "${TQ_TEXT_PACKAGE_DIR}")
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/TQText")
install(FILES README.md DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/TQText")
