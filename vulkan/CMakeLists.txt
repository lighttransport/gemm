# Vulkan Cooperative Matrix Matmul Benchmark
#
# Self-contained benchmark targeting AMD RDNA4 (gfx1200+)
# with VK_KHR_cooperative_matrix extension.
#
# Uses bundled vkew for runtime Vulkan loading (no link-time dependency).
#
cmake_minimum_required(VERSION 3.16)
project(vulkan_coopmat_matmul C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find glslc compiler (required for shader compilation)
find_program(GLSLC_EXECUTABLE
    NAMES glslc
    HINTS
        $ENV{VULKAN_SDK}/bin
        /usr/bin
        /usr/local/bin
)

if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found. Please install Vulkan SDK.")
else()
    message(STATUS "Found glslc: ${GLSLC_EXECUTABLE}")
endif()

# Shader directories
set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

# List of shaders to compile
set(SHADER_SOURCES
    ${SHADER_SOURCE_DIR}/matmul_coopmat_f16.comp
    ${SHADER_SOURCE_DIR}/matmul_tiled_f16.comp
    ${SHADER_SOURCE_DIR}/matmul_naive_f32.comp
    ${SHADER_SOURCE_DIR}/add_f32.comp
    ${SHADER_SOURCE_DIR}/gelu_f32.comp
    ${SHADER_SOURCE_DIR}/matmul_bias_f32.comp
    ${SHADER_SOURCE_DIR}/layernorm_f32.comp
    ${SHADER_SOURCE_DIR}/softmax_f32.comp
    ${SHADER_SOURCE_DIR}/patch_embed_f32.comp
    ${SHADER_SOURCE_DIR}/add_transposed_f32.comp
    ${SHADER_SOURCE_DIR}/spatial_merge_f32.comp
    ${SHADER_SOURCE_DIR}/attn_naive_f32.comp
    ${SHADER_SOURCE_DIR}/attn_flash_f32.comp
)

# Custom function to compile GLSL to SPIR-V
function(compile_shader SHADER_SOURCE)
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME_WE)
    set(SPIRV_OUTPUT ${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv)

    # Check if this is a cooperative matrix shader
    string(FIND ${SHADER_NAME} "coopmat" IS_COOPMAT)
    if(IS_COOPMAT GREATER -1)
        # Cooperative matrix requires Vulkan 1.3+ and SPIR-V 1.6
        add_custom_command(
            OUTPUT ${SPIRV_OUTPUT}
            COMMAND ${GLSLC_EXECUTABLE}
                    -fshader-stage=compute
                    --target-env=vulkan1.3
                    --target-spv=spv1.6
                    -O
                    -o ${SPIRV_OUTPUT}
                    ${SHADER_SOURCE}
            DEPENDS ${SHADER_SOURCE}
            COMMENT "Compiling ${SHADER_NAME}.comp (cooperative matrix) -> ${SHADER_NAME}.spv"
            VERBATIM
        )
    else()
        add_custom_command(
            OUTPUT ${SPIRV_OUTPUT}
            COMMAND ${GLSLC_EXECUTABLE}
                    -fshader-stage=compute
                    -O
                    -o ${SPIRV_OUTPUT}
                    ${SHADER_SOURCE}
            DEPENDS ${SHADER_SOURCE}
            COMMENT "Compiling ${SHADER_NAME}.comp -> ${SHADER_NAME}.spv"
            VERBATIM
        )
    endif()
    list(APPEND SPIRV_OUTPUTS ${SPIRV_OUTPUT})
    set(SPIRV_OUTPUTS ${SPIRV_OUTPUTS} PARENT_SCOPE)
endfunction()

# Compile all shaders
set(SPIRV_OUTPUTS "")
foreach(SHADER ${SHADER_SOURCES})
    if(EXISTS ${SHADER})
        compile_shader(${SHADER})
    else()
        message(WARNING "Shader not found: ${SHADER}")
    endif()
endforeach()

# Custom target for shader compilation
if(SPIRV_OUTPUTS)
    add_custom_target(coopmat_shaders ALL DEPENDS ${SPIRV_OUTPUTS})
endif()

# Benchmark executable using local deps
set(BENCHMARK_TARGET matmul_vulkan_coopmat)
set(DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps)

add_executable(${BENCHMARK_TARGET}
    test_matmul_coopmat.cc
    ${DEPS_DIR}/vulkan-runner.cc
    ${DEPS_DIR}/vkew.cc
)

target_include_directories(${BENCHMARK_TARGET} PRIVATE
    ${DEPS_DIR}
)

# vkew dynamically loads Vulkan at runtime - only need dl library
target_link_libraries(${BENCHMARK_TARGET} PRIVATE
    ${CMAKE_DL_LIBS}
)

target_compile_definitions(${BENCHMARK_TARGET} PRIVATE
    USE_VULKAN=1
)

# Ensure shaders are built before the benchmark
if(TARGET coopmat_shaders)
    add_dependencies(${BENCHMARK_TARGET} coopmat_shaders)
endif()

# Copy shaders to binary directory for easy access
if(SPIRV_OUTPUTS)
    add_custom_command(TARGET ${BENCHMARK_TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${SPIRV_OUTPUTS}
                ${CMAKE_BINARY_DIR}/shaders/
        COMMENT "Copying compiled shaders to build directory"
    )
endif()

# Vision encoder test executable
set(VISION_TARGET test_vision_encoder)

add_executable(${VISION_TARGET}
    test_vision_encoder.cc
    vulkan_vision_encoder.cc
    vision_cpu_impl.c
    ${DEPS_DIR}/vulkan-runner.cc
    ${DEPS_DIR}/vkew.cc
)

target_include_directories(${VISION_TARGET} PRIVATE
    ${DEPS_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(${VISION_TARGET} PRIVATE
    ${CMAKE_DL_LIBS}
    m
)

target_compile_definitions(${VISION_TARGET} PRIVATE
    USE_VULKAN=1
)

if(TARGET coopmat_shaders)
    add_dependencies(${VISION_TARGET} coopmat_shaders)
endif()

if(SPIRV_OUTPUTS)
    add_custom_command(TARGET ${VISION_TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${SPIRV_OUTPUTS}
                ${CMAKE_BINARY_DIR}/shaders/
        COMMENT "Copying compiled shaders to build directory (vision)"
    )
endif()

# Multimodal inference executable (GPU vision + CPU LLM)
set(MULTIMODAL_TARGET test_vision_multimodal)

add_executable(${MULTIMODAL_TARGET}
    test_vision_multimodal.cc
    vulkan_vision_encoder.cc
    vision_cpu_impl.c
    ${DEPS_DIR}/vulkan-runner.cc
    ${DEPS_DIR}/vkew.cc
)

target_include_directories(${MULTIMODAL_TARGET} PRIVATE
    ${DEPS_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(${MULTIMODAL_TARGET} PRIVATE
    ${CMAKE_DL_LIBS}
    m
)

target_compile_definitions(${MULTIMODAL_TARGET} PRIVATE
    USE_VULKAN=1
)

if(TARGET coopmat_shaders)
    add_dependencies(${MULTIMODAL_TARGET} coopmat_shaders)
endif()

if(SPIRV_OUTPUTS)
    add_custom_command(TARGET ${MULTIMODAL_TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${SPIRV_OUTPUTS}
                ${CMAKE_BINARY_DIR}/shaders/
        COMMENT "Copying compiled shaders to build directory (multimodal)"
    )
endif()

message(STATUS "")
message(STATUS "Vulkan Cooperative Matrix Matmul Benchmark")
message(STATUS "  Shader source: ${SHADER_SOURCE_DIR}")
message(STATUS "  Shader output: ${SHADER_OUTPUT_DIR}")
message(STATUS "  Dependencies:  ${DEPS_DIR}")
message(STATUS "")
