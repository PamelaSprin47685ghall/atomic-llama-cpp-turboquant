add_executable(tp5-logits tp5-logits.cpp)
target_link_libraries(tp5-logits PRIVATE llama-common)
target_compile_features(tp5-logits PRIVATE cxx_std_17)
