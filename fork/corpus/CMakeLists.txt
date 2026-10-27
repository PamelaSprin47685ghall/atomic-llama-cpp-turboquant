cmake_minimum_required(VERSION 3.16)
project(TQCorpus VERSION 1.0.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(tq_corpus src/trie.cpp)
add_library(TQCorpus::corpus ALIAS tq_corpus)
set_target_properties(tq_corpus PROPERTIES EXPORT_NAME corpus)
target_compile_features(tq_corpus PUBLIC cxx_std_17)
target_include_directories(tq_corpus PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

include(CTest)
if(BUILD_TESTING)
    add_executable(tq-corpus-test tests/trie_test.cpp)
    target_link_libraries(tq-corpus-test PRIVATE TQCorpus::corpus)
    add_test(NAME tq-corpus COMMAND tq-corpus-test)
endif()

install(TARGETS tq_corpus EXPORT TQCorpusTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT TQCorpusTargets
    FILE TQCorpusTargets.cmake
    NAMESPACE TQCorpus::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TQCorpus)

configure_package_config_file(cmake/TQCorpusConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/TQCorpusConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TQCorpus)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/TQCorpusConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TQCorpusConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TQCorpusConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TQCorpus)
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/TQCorpus)
