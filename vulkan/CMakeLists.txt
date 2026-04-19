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

find_package(Threads REQUIRED)

# Enable AVX2 + F16C + FMA for SIMD-accelerated F16 dot products on x86_64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -mfma -mf16c")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -mf16c")
endif()

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
    ${SHADER_SOURCE_DIR}/rmsnorm_f32.comp
    ${SHADER_SOURCE_DIR}/matvec_f32.comp
    ${SHADER_SOURCE_DIR}/matvec_f16_f32.comp
    ${SHADER_SOURCE_DIR}/silu_mul_f32.comp
    ${SHADER_SOURCE_DIR}/rope_neox_f32.comp
    ${SHADER_SOURCE_DIR}/rope_mrope_f32.comp
    ${SHADER_SOURCE_DIR}/rope_vision_f32.comp
    ${SHADER_SOURCE_DIR}/qknorm_f32.comp
    ${SHADER_SOURCE_DIR}/kv_cache_store_f32.comp
    ${SHADER_SOURCE_DIR}/attn_decode_f32.comp
    ${SHADER_SOURCE_DIR}/matvec_q8_0_f32.comp
    ${SHADER_SOURCE_DIR}/dequant_q8_0_f16.comp
    ${SHADER_SOURCE_DIR}/matmul_coopmat_nt_f16.comp
    ${SHADER_SOURCE_DIR}/rope_neox_batch_f32.comp
    ${SHADER_SOURCE_DIR}/rope_mrope_batch_f32.comp
    ${SHADER_SOURCE_DIR}/kv_cache_store_batch_f32.comp
    ${SHADER_SOURCE_DIR}/attn_prefill_f32.comp
    ${SHADER_SOURCE_DIR}/silu_f32.comp
)

# DA3 depth estimation shaders
set(DA3_SHADER_SOURCES
    ${SHADER_SOURCE_DIR}/da3/resize_normalize_f32.comp
    ${SHADER_SOURCE_DIR}/da3/cls_pos_embed_f32.comp
    ${SHADER_SOURCE_DIR}/da3/rope_2d_f32.comp
    ${SHADER_SOURCE_DIR}/da3/swiglu_f32.comp
    ${SHADER_SOURCE_DIR}/da3/kv_transpose_f32.comp
    ${SHADER_SOURCE_DIR}/da3/flash_attn_tiled_f32.comp
    ${SHADER_SOURCE_DIR}/da3/cat_local_global_f32.comp
    ${SHADER_SOURCE_DIR}/da3/dpt_cls_concat_f32.comp
    ${SHADER_SOURCE_DIR}/da3/strided_layernorm_f32.comp
    ${SHADER_SOURCE_DIR}/da3/dpt_tok_to_chw_f32.comp
    ${SHADER_SOURCE_DIR}/da3/deconv_scatter_f32.comp
    ${SHADER_SOURCE_DIR}/da3/conv2d_f32.comp
    ${SHADER_SOURCE_DIR}/da3/bilinear_upsample_f32.comp
    ${SHADER_SOURCE_DIR}/da3/depth_activation_f32.comp
    ${SHADER_SOURCE_DIR}/da3/sinusoidal_uv_posembed_f32.comp
    ${SHADER_SOURCE_DIR}/da3/relu_f32.comp
    ${SHADER_SOURCE_DIR}/da3/channel_layernorm_f32.comp
    ${SHADER_SOURCE_DIR}/da3/qk_layernorm_offset_f32.comp
    ${SHADER_SOURCE_DIR}/da3/patch_embed_conv2d_f32.comp
    ${SHADER_SOURCE_DIR}/da3/copy_f32.comp
    ${SHADER_SOURCE_DIR}/da3/silu_f32.comp
    ${SHADER_SOURCE_DIR}/da3/matmul_bias_f16_f32.comp
)

# TRELLIS2 shaders
set(T2_SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders/trellis2)
set(T2_SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders/trellis2)
file(MAKE_DIRECTORY ${T2_SHADER_OUTPUT_DIR})

set(T2_SHADER_SOURCES
    ${T2_SHADER_SOURCE_DIR}/adaln_f32.comp
    ${T2_SHADER_SOURCE_DIR}/gated_add_f32.comp
    ${T2_SHADER_SOURCE_DIR}/modulation_f32.comp
    ${T2_SHADER_SOURCE_DIR}/rope_3d_f32.comp
    ${T2_SHADER_SOURCE_DIR}/timestep_embed_cossin_f32.comp
    ${T2_SHADER_SOURCE_DIR}/split_qkv_chunk_f32.comp
    ${T2_SHADER_SOURCE_DIR}/split_kv_chunk_f32.comp
    ${T2_SHADER_SOURCE_DIR}/layernorm_noaffine_f32.comp
    ${T2_SHADER_SOURCE_DIR}/silu_inplace_f32.comp
    ${T2_SHADER_SOURCE_DIR}/conv3d_k3_f32.comp
    ${T2_SHADER_SOURCE_DIR}/channel_layernorm_3d_f32.comp
    ${T2_SHADER_SOURCE_DIR}/pixel_shuffle_3d_f32.comp
    ${T2_SHADER_SOURCE_DIR}/self_attn_tiled_f32.comp
    ${T2_SHADER_SOURCE_DIR}/sparse_conv3d_f32.comp
    ${T2_SHADER_SOURCE_DIR}/rope_2d_dinov3_f32.comp
    ${T2_SHADER_SOURCE_DIR}/dinov3_prepend_tokens_f32.comp
)

# HY3D shaders (reused by DA3 and TRELLIS2)
set(HY3D_SHADER_SOURCES
    ${SHADER_SOURCE_DIR}/hy3d/layerscale_add_f32.comp
    ${SHADER_SOURCE_DIR}/hy3d/qk_layernorm_f32.comp
    ${SHADER_SOURCE_DIR}/hy3d/gemm_f32_f32.comp
    ${SHADER_SOURCE_DIR}/hy3d/rms_norm_f32.comp
    ${SHADER_SOURCE_DIR}/hy3d/cross_attn_f32.comp
    ${SHADER_SOURCE_DIR}/hy3d/cfg_combine_f32.comp
    ${SHADER_SOURCE_DIR}/hy3d/euler_step_f32.comp
)

# Flux.2 Klein text-to-image shaders
set(FLUX2_SHADER_SOURCES
    ${SHADER_SOURCE_DIR}/flux2_adaln_f32.comp
    ${SHADER_SOURCE_DIR}/flux2_rmsnorm_per_head_f32.comp
    ${SHADER_SOURCE_DIR}/flux2_rope_img_f32.comp
    ${SHADER_SOURCE_DIR}/flux2_rope_txt_f32.comp
    ${SHADER_SOURCE_DIR}/flux2_swiglu_f32.comp
    ${SHADER_SOURCE_DIR}/flux2_gated_add_f32.comp
    ${SHADER_SOURCE_DIR}/flux2_bf16_trunc_f32.comp
    ${SHADER_SOURCE_DIR}/flux2_gemm_f32.comp
    ${SHADER_SOURCE_DIR}/flux2_attn_flash_f32.comp
    ${SHADER_SOURCE_DIR}/vae_conv2d_3x3_f32.comp
    ${SHADER_SOURCE_DIR}/vae_conv2d_1x1_f32.comp
    ${SHADER_SOURCE_DIR}/vae_groupnorm_silu_f32.comp
    ${SHADER_SOURCE_DIR}/vae_upsample2x_f32.comp
    ${SHADER_SOURCE_DIR}/vae_transpose_f32.comp
    ${SHADER_SOURCE_DIR}/vae_add_bias_f32.comp
)

# Qwen-Image text-to-image shaders
set(QIMG_SHADER_SOURCES
    ${SHADER_SOURCE_DIR}/qimg_rope_2d_f32.comp
    ${SHADER_SOURCE_DIR}/qimg_rope_1d_f32.comp
    ${SHADER_SOURCE_DIR}/qimg_patchify_f32.comp
    ${SHADER_SOURCE_DIR}/qimg_unpatchify_f32.comp
    ${SHADER_SOURCE_DIR}/qimg_euler_step_f32.comp
    ${SHADER_SOURCE_DIR}/qimg_cfg_combine_f32.comp
    ${SHADER_SOURCE_DIR}/qimg_rmsnorm_weighted_f32.comp
    ${SHADER_SOURCE_DIR}/qimg_quantize_fp8_rt_f32.comp
    ${SHADER_SOURCE_DIR}/qimg_vae_conv2d_f32.comp
    ${SHADER_SOURCE_DIR}/qimg_vae_rmsnorm_f32.comp
    ${SHADER_SOURCE_DIR}/qimg_vae_silu_f32.comp
    ${SHADER_SOURCE_DIR}/qimg_vae_upsample2x_f32.comp
)

# Custom function to compile GLSL to SPIR-V (supports subdirectories)
function(compile_shader_subdir SHADER_SOURCE SUBDIR)
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME_WE)
    file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR}/${SUBDIR})
    set(SPIRV_OUTPUT ${SHADER_OUTPUT_DIR}/${SUBDIR}/${SHADER_NAME}.spv)
    add_custom_command(
        OUTPUT ${SPIRV_OUTPUT}
        COMMAND ${GLSLC_EXECUTABLE} -fshader-stage=compute -O -o ${SPIRV_OUTPUT} ${SHADER_SOURCE}
        DEPENDS ${SHADER_SOURCE}
        COMMENT "Compiling ${SUBDIR}/${SHADER_NAME}.comp -> ${SUBDIR}/${SHADER_NAME}.spv"
        VERBATIM
    )
    list(APPEND SPIRV_OUTPUTS ${SPIRV_OUTPUT})
    set(SPIRV_OUTPUTS ${SPIRV_OUTPUTS} PARENT_SCOPE)
endfunction()

# Custom function to compile GLSL to SPIR-V
function(compile_shader SHADER_SOURCE)
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME_WE)
    set(SPIRV_OUTPUT ${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv)

    # Check if this shader needs Vulkan 1.3+ (coopmat or subgroup ops)
    string(FIND ${SHADER_NAME} "coopmat" IS_COOPMAT)
    string(FIND ${SHADER_NAME} "matvec" IS_MATVEC)
    if(IS_COOPMAT GREATER -1 OR IS_MATVEC GREATER -1)
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

# Compile DA3 shaders
foreach(SHADER ${DA3_SHADER_SOURCES})
    if(EXISTS ${SHADER})
        compile_shader_subdir(${SHADER} "da3")
    else()
        message(WARNING "DA3 shader not found: ${SHADER}")
    endif()
endforeach()

# Compile HY3D shaders (needed by DA3 for layerscale_add, qk_layernorm)
foreach(SHADER ${HY3D_SHADER_SOURCES})
    if(EXISTS ${SHADER})
        compile_shader_subdir(${SHADER} "hy3d")
    else()
        message(WARNING "HY3D shader not found: ${SHADER}")
    endif()
endforeach()

# Compile TRELLIS2 shaders
set(T2_SPIRV_OUTPUTS "")
foreach(SHADER ${T2_SHADER_SOURCES})
    if(EXISTS ${SHADER})
        get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
        set(SPIRV_OUTPUT ${T2_SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv)
        add_custom_command(
            OUTPUT ${SPIRV_OUTPUT}
            COMMAND ${GLSLC_EXECUTABLE}
                    -fshader-stage=compute
                    -O
                    -o ${SPIRV_OUTPUT}
                    ${SHADER}
            DEPENDS ${SHADER}
            COMMENT "Compiling trellis2/${SHADER_NAME}.comp -> ${SHADER_NAME}.spv"
            VERBATIM
        )
        list(APPEND T2_SPIRV_OUTPUTS ${SPIRV_OUTPUT})
    endif()
endforeach()

if(T2_SPIRV_OUTPUTS)
    add_custom_target(trellis2_shaders ALL DEPENDS ${T2_SPIRV_OUTPUTS})
endif()

# Compile Flux2 shaders
foreach(SHADER ${FLUX2_SHADER_SOURCES})
    if(EXISTS ${SHADER})
        compile_shader(${SHADER})
    else()
        message(WARNING "Flux2 shader not found: ${SHADER}")
    endif()
endforeach()

# Compile Qwen-Image shaders
foreach(SHADER ${QIMG_SHADER_SOURCES})
    if(EXISTS ${SHADER})
        compile_shader(${SHADER})
    else()
        message(WARNING "Qwen-Image shader not found: ${SHADER}")
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
    PROF_DISABLED=1
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
    vulkan_llm_runner.cc
    vision_cpu_impl.c
    stb_impl.c
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
    Threads::Threads
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

# Vulkan LLM test executable
set(LLM_TARGET test_vulkan_llm)

add_executable(${LLM_TARGET}
    test_vulkan_llm.cc
    vulkan_llm_runner.cc
    vision_cpu_impl.c
    ${DEPS_DIR}/vulkan-runner.cc
    ${DEPS_DIR}/vkew.cc
)

target_include_directories(${LLM_TARGET} PRIVATE
    ${DEPS_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(${LLM_TARGET} PRIVATE
    ${CMAKE_DL_LIBS}
    m
    Threads::Threads
)

target_compile_definitions(${LLM_TARGET} PRIVATE
    USE_VULKAN=1
    PROF_DISABLED=1
)

if(TARGET coopmat_shaders)
    add_dependencies(${LLM_TARGET} coopmat_shaders)
endif()

if(SPIRV_OUTPUTS)
    add_custom_command(TARGET ${LLM_TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${SPIRV_OUTPUTS}
                ${CMAKE_BINARY_DIR}/shaders/
        COMMENT "Copying compiled shaders to build directory (llm)"
    )
endif()

# DA3 Depth Estimation test executable
set(DA3_TARGET test_vulkan_da3)

add_executable(${DA3_TARGET}
    da3/test_vulkan_da3.cc
    da3/vulkan_da3_runner.cc
    da3/stb_impl.c
    ${DEPS_DIR}/vulkan-runner.cc
    ${DEPS_DIR}/vkew.cc
)

target_include_directories(${DA3_TARGET} PRIVATE
    ${DEPS_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/da3
)

target_link_libraries(${DA3_TARGET} PRIVATE
    ${CMAKE_DL_LIBS}
    m
    Threads::Threads
)

target_compile_definitions(${DA3_TARGET} PRIVATE
    USE_VULKAN=1
)

if(TARGET coopmat_shaders)
    add_dependencies(${DA3_TARGET} coopmat_shaders)
endif()

# TRELLIS2 Vulkan test executable
set(TRELLIS2_VK_TARGET test_vulkan_trellis2)

add_executable(${TRELLIS2_VK_TARGET}
    trellis2/test_vulkan_trellis2.cc
    trellis2/vulkan_trellis2_runner.cc
    ${DEPS_DIR}/vulkan-runner.cc
    ${DEPS_DIR}/vkew.cc
)

target_include_directories(${TRELLIS2_VK_TARGET} PRIVATE
    ${DEPS_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(${TRELLIS2_VK_TARGET} PRIVATE
    ${CMAKE_DL_LIBS}
    m
)

target_compile_definitions(${TRELLIS2_VK_TARGET} PRIVATE
    USE_VULKAN=1
)

if(TARGET trellis2_shaders)
    add_dependencies(${TRELLIS2_VK_TARGET} trellis2_shaders)
endif()
if(TARGET coopmat_shaders)
    add_dependencies(${TRELLIS2_VK_TARGET} coopmat_shaders)
endif()

# Flux.2 Klein text-to-image test executable
set(FLUX2_TARGET test_vulkan_flux2)

add_executable(${FLUX2_TARGET}
    flux2/test_vulkan_flux2.cc
    flux2/vulkan_flux2_runner.cc
    flux2/ggml_dequant_impl.c
    ${DEPS_DIR}/vulkan-runner.cc
    ${DEPS_DIR}/vkew.cc
)

target_include_directories(${FLUX2_TARGET} PRIVATE
    ${DEPS_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/flux2
)

target_link_libraries(${FLUX2_TARGET} PRIVATE
    ${CMAKE_DL_LIBS}
    m
    Threads::Threads
)

target_compile_definitions(${FLUX2_TARGET} PRIVATE
    USE_VULKAN=1
)

if(TARGET coopmat_shaders)
    add_dependencies(${FLUX2_TARGET} coopmat_shaders)
endif()

# Qwen-Image text-to-image test executable
set(QIMG_TARGET test_vulkan_qimg)

add_executable(${QIMG_TARGET}
    qimg/test_vulkan_qimg.cc
    qimg/vulkan_qimg_runner.cc
    qimg/ggml_dequant_impl.c
    ${DEPS_DIR}/vulkan-runner.cc
    ${DEPS_DIR}/vkew.cc
)

target_include_directories(${QIMG_TARGET} PRIVATE
    ${DEPS_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/qimg
)

target_link_libraries(${QIMG_TARGET} PRIVATE
    ${CMAKE_DL_LIBS}
    m
    Threads::Threads
)

target_compile_definitions(${QIMG_TARGET} PRIVATE
    USE_VULKAN=1
)

if(TARGET coopmat_shaders)
    add_dependencies(${QIMG_TARGET} coopmat_shaders)
endif()

message(STATUS "")
message(STATUS "Vulkan Cooperative Matrix Matmul Benchmark")
message(STATUS "  Shader source: ${SHADER_SOURCE_DIR}")
message(STATUS "  Shader output: ${SHADER_OUTPUT_DIR}")
message(STATUS "  Dependencies:  ${DEPS_DIR}")
message(STATUS "")
