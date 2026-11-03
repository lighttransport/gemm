cmake_minimum_required(VERSION 3.20)
project(vulkan_gemm C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

find_program(SHADER_COMPILER NAMES glslangValidator glslc
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
    "C:/Program Files/RenderDoc/plugins/spirv"
    DOC "Path to glslangValidator or glslc")
if(NOT SHADER_COMPILER)
    message(FATAL_ERROR "Set -DSHADER_COMPILER=path/to/glslangValidator (or glslc)")
endif()
get_filename_component(SHADER_COMPILER_NAME "${SHADER_COMPILER}" NAME)
if(SHADER_COMPILER_NAME MATCHES "^glslc")
    set(SHADER_FLAGS --target-env=vulkan1.3 -O)
else()
    set(SHADER_FLAGS -V --target-env vulkan1.3)
endif()

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(SHADER_OUTPUTS)
foreach(TYPE RANGE 0 5)
    set(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shaders/gemm_${TYPE}.spv")
    add_custom_command(OUTPUT "${OUTPUT}"
        COMMAND "${SHADER_COMPILER}" ${SHADER_FLAGS} -DTYPE=${TYPE}
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/gemm.comp" -o "${OUTPUT}"
        DEPENDS shaders/gemm.comp VERBATIM)
    list(APPEND SHADER_OUTPUTS "${OUTPUT}")
endforeach()
foreach(SHADER vram gemm_fp32_register pack_fp32)
    set(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER}.spv")
    add_custom_command(OUTPUT "${OUTPUT}"
        COMMAND "${SHADER_COMPILER}" ${SHADER_FLAGS}
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER}.comp" -o "${OUTPUT}"
        DEPENDS "shaders/${SHADER}.comp" VERBATIM)
    list(APPEND SHADER_OUTPUTS "${OUTPUT}")
endforeach()
add_custom_target(shaders ALL DEPENDS ${SHADER_OUTPUTS})

add_executable(bench_vulkan_gemm bench_vulkan_gemm.c ../../vulkan/deps/vkew.c)
target_include_directories(bench_vulkan_gemm PRIVATE ../../vulkan/deps)
target_compile_definitions(bench_vulkan_gemm PRIVATE
    SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders" _CRT_SECURE_NO_WARNINGS)
if(MSVC)
    target_compile_options(bench_vulkan_gemm PRIVATE /W4)
else()
    target_compile_options(bench_vulkan_gemm PRIVATE -Wall -Wextra -Wpedantic)
    if(NOT WIN32)
        target_link_libraries(bench_vulkan_gemm PRIVATE m ${CMAKE_DL_LIBS})
    endif()
endif()
add_dependencies(bench_vulkan_gemm shaders)
