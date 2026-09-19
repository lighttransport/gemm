cmake_minimum_required(VERSION 3.16)
project(glm53f_boundary_compare LANGUAGES C CXX)

if (NOT DEFINED LLAMA_SRC)
    message(FATAL_ERROR "LLAMA_SRC must point to the llama.cpp checkout")
endif()

find_package(Threads REQUIRED)
add_subdirectory("${LLAMA_SRC}" "${CMAKE_BINARY_DIR}/llama" EXCLUDE_FROM_ALL)

add_executable(glm53f_embedding_compare
    ../glm53f_embedding_compare.cpp)
target_include_directories(glm53f_embedding_compare PRIVATE
    "${LLAMA_SRC}/include"
    "${LLAMA_SRC}/src"
    "${LLAMA_SRC}/ggml/include"
    "${LLAMA_SRC}/common"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../common")
target_link_libraries(glm53f_embedding_compare PRIVATE
    llama-common llama Threads::Threads)
target_compile_features(glm53f_embedding_compare PRIVATE cxx_std_17)
