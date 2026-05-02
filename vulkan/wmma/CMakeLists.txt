cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(vulkan_wmma_gemm CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    find_package(Threads REQUIRED)
    set(DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../deps)
endif()

find_program(GLSLC_EXECUTABLE
    NAMES glslc
    HINTS $ENV{VULKAN_SDK}/bin /usr/bin /usr/local/bin
)

if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found. Please install Vulkan SDK.")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(WMMA_SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(WMMA_SHADER_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders/wmma)
file(MAKE_DIRECTORY ${WMMA_SHADER_OUTPUT_DIR})

set(WMMA_SPV ${WMMA_SHADER_OUTPUT_DIR}/gemm_wmma_f16.spv)
set(WMMA_BF16_SPV ${WMMA_SHADER_OUTPUT_DIR}/gemm_wmma_bf16.spv)
add_custom_command(
    OUTPUT ${WMMA_SPV}
    COMMAND ${GLSLC_EXECUTABLE}
            -fshader-stage=compute
            --target-env=vulkan1.3
            --target-spv=spv1.6
            -O
            -o ${WMMA_SPV}
            ${WMMA_SHADER_SOURCE_DIR}/gemm_wmma_f16.comp
    DEPENDS ${WMMA_SHADER_SOURCE_DIR}/gemm_wmma_f16.comp
    COMMENT "Compiling wmma/gemm_wmma_f16.comp"
    VERBATIM
)

add_custom_command(
    OUTPUT ${WMMA_BF16_SPV}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/patch_spv_bf16.py ${WMMA_SPV} ${WMMA_BF16_SPV}
    DEPENDS ${WMMA_SPV} ${CMAKE_CURRENT_SOURCE_DIR}/patch_spv_bf16.py
    COMMENT "Patching wmma/gemm_wmma_f16.spv -> BF16"
    VERBATIM
)

add_custom_target(wmma_shaders ALL DEPENDS ${WMMA_SPV} ${WMMA_BF16_SPV})

add_executable(bench_wmma_gemm
    bench_wmma_gemm.cc
    ${DEPS_DIR}/vulkan-runner.cc
    ${DEPS_DIR}/vkew.cc
)

target_include_directories(bench_wmma_gemm PRIVATE ${DEPS_DIR})
target_compile_definitions(bench_wmma_gemm PRIVATE USE_VULKAN=1)
target_link_libraries(bench_wmma_gemm PRIVATE ${CMAKE_DL_LIBS} m Threads::Threads)
add_dependencies(bench_wmma_gemm wmma_shaders)
