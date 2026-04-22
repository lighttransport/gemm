cmake_minimum_required(VERSION 3.10)
project(diffusion_server C)

option(DIFFUSION_SERVER_ENABLE_QWEN_IMAGE "Enable Qwen-Image model runner" ON)
option(DIFFUSION_SERVER_ENABLE_QWEN_IMAGE_CPU "Enable Qwen-Image CPU backend" ON)
option(DIFFUSION_SERVER_ENABLE_QWEN_IMAGE_HIP "Enable Qwen-Image HIP backend" ON)
option(DIFFUSION_SERVER_ENABLE_SAM3 "Enable SAM3 API surface" ON)
option(DIFFUSION_SERVER_ENABLE_SAM3_CUDA "Enable SAM3 CUDA backend (NVRTC via cuew)" OFF)
option(DIFFUSION_SERVER_ENABLE_MCP "Enable minimal MCP stdio mode" OFF)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Default to Release if the user didn't pick one. Otherwise CMake leaves
# CMAKE_C_FLAGS empty -> gcc compiles at -O0, which makes the SAM3 / Qwen
# CPU kernels ~3x slower than the standalone Makefiles (which use -O3).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -funroll-loops" CACHE STRING "" FORCE)

add_executable(diffusion-server server.c)
target_include_directories(diffusion-server PRIVATE
    ${ROOT_DIR}/common
    ${ROOT_DIR}/rdna4
    ${ROOT_DIR}/rdna4/llm
    ${ROOT_DIR}/rdna4/qimg)

target_compile_definitions(diffusion-server PRIVATE
    $<$<BOOL:${DIFFUSION_SERVER_ENABLE_QWEN_IMAGE}>:DIFFUSION_SERVER_ENABLE_QWEN_IMAGE=1>
    $<$<BOOL:${DIFFUSION_SERVER_ENABLE_QWEN_IMAGE_CPU}>:DIFFUSION_SERVER_ENABLE_QWEN_IMAGE_CPU=1>
    $<$<BOOL:${DIFFUSION_SERVER_ENABLE_QWEN_IMAGE_HIP}>:DIFFUSION_SERVER_ENABLE_QWEN_IMAGE_HIP=1>
    $<$<BOOL:${DIFFUSION_SERVER_ENABLE_SAM3}>:DIFFUSION_SERVER_ENABLE_SAM3=1>
    $<$<BOOL:${DIFFUSION_SERVER_ENABLE_SAM3_CUDA}>:DIFFUSION_SERVER_ENABLE_SAM3_CUDA=1>
    $<$<BOOL:${DIFFUSION_SERVER_ENABLE_MCP}>:DIFFUSION_SERVER_ENABLE_MCP=1>)

target_compile_options(diffusion-server PRIVATE -Wall -Wextra -Wno-unused-function -Wno-unused-parameter)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(diffusion-server PRIVATE -mavx2 -mfma -mf16c)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    target_compile_options(diffusion-server PRIVATE -march=armv8.2-a+sve -ffp-contract=fast)
endif()

target_link_libraries(diffusion-server PRIVATE m dl pthread)

if(DIFFUSION_SERVER_ENABLE_QWEN_IMAGE_HIP)
    target_sources(diffusion-server PRIVATE
        ${ROOT_DIR}/rdna4/qimg/hip_qimg_runner.c
        ${ROOT_DIR}/rdna4/llm/hip_llm_runner.c
        ${ROOT_DIR}/rdna4/rocew.c)
endif()

if(DIFFUSION_SERVER_ENABLE_SAM3)
    # sam3 CPU runner — compiled as separate TUs so their _IMPLEMENTATION
    # macros do not collide with server.c's.
    target_sources(diffusion-server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/server_sam3.c
        ${ROOT_DIR}/cpu/sam3/sam3_runner.c
        ${ROOT_DIR}/cpu/sam3/sam3_clip_bpe.c)
    target_include_directories(diffusion-server PRIVATE
        ${ROOT_DIR}/cpu/sam3)
    # server.c already carries SAFETENSORS / GGML_DEQUANT implementations —
    # tell sam3_runner.c to use declarations only, to avoid multiple-definition
    # link errors. The standalone cpu/sam3 build is unaffected.
    set_source_files_properties(${ROOT_DIR}/cpu/sam3/sam3_runner.c
        PROPERTIES COMPILE_DEFINITIONS "SAM3_RUNNER_EXTERNAL_IMPLS=1")
    # sam3_runner.c includes "safetensors.h", "stb_image_resize2.h",
    # "ggml_dequant.h", "cpu_compute.h" via its local include path — common/
    # is already on the include list (above), so these resolve.
    # OpenMP for sam3 CPU kernels (optional but speeds up ViT massively).
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        target_link_libraries(diffusion-server PRIVATE OpenMP::OpenMP_C)
    endif()
endif()

if(DIFFUSION_SERVER_ENABLE_SAM3_CUDA)
    if(NOT DIFFUSION_SERVER_ENABLE_SAM3)
        message(FATAL_ERROR "DIFFUSION_SERVER_ENABLE_SAM3_CUDA requires DIFFUSION_SERVER_ENABLE_SAM3")
    endif()
    # CUDA sam3 runner — uses NVRTC via cuew (no nvcc). Compiled as a
    # separate TU; SAFETENSORS_IMPLEMENTATION is suppressed via
    # CUDA_SAM3_RUNNER_EXTERNAL_IMPLS to avoid clashing with server.c.
    target_sources(diffusion-server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/server_sam3_cuda.c
        ${ROOT_DIR}/cuda/sam3/cuda_sam3_runner.c
        ${ROOT_DIR}/cuda/cuew.c)
    target_include_directories(diffusion-server PRIVATE
        ${ROOT_DIR}/cuda
        ${ROOT_DIR}/cuda/sam3)
    set_source_files_properties(${ROOT_DIR}/cuda/sam3/cuda_sam3_runner.c
        PROPERTIES COMPILE_DEFINITIONS "CUDA_SAM3_RUNNER_EXTERNAL_IMPLS=1")
endif()
