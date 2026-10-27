cmake_minimum_required(VERSION 3.16)
project(TQCompose VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

if(NOT TARGET llama)
    find_package(llama CONFIG REQUIRED)
endif()
if(NOT TARGET TQSampling::sampling)
    find_package(TQSampling CONFIG REQUIRED)
endif()

add_library(tq-compose STATIC src/sampler.cpp)
add_library(TQCompose::compose ALIAS tq-compose)
set_target_properties(tq-compose PROPERTIES EXPORT_NAME compose POSITION_INDEPENDENT_CODE ON)
target_compile_features(tq-compose PUBLIC cxx_std_17)
target_include_directories(tq-compose PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(tq-compose PRIVATE llama TQSampling::sampling)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tq-compose PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_executable(llama-compose src/main.cpp)
set_target_properties(llama-compose PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
target_link_libraries(llama-compose PRIVATE tq-compose llama)
target_compile_features(llama-compose PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(llama-compose PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(BUILD_TESTING)
    add_executable(test-tq-compose tests/test-sampler.cpp)
    target_link_libraries(test-tq-compose PRIVATE tq-compose llama)
    add_test(NAME tq-compose-sampler COMMAND test-tq-compose)
    add_test(NAME tq-compose-help COMMAND llama-compose --help)
endif()

install(TARGETS tq-compose EXPORT TQComposeTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS llama-compose RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/TQCompose)
install(EXPORT TQComposeTargets FILE TQComposeTargets.cmake NAMESPACE TQCompose::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TQCompose)
configure_package_config_file(cmake/TQComposeConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/TQComposeConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TQCompose)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/TQComposeConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/TQComposeConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TQComposeConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TQCompose)
