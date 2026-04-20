cmake_minimum_required(VERSION 3.10)
project(diffusion_server C)

option(DIFFUSION_SERVER_ENABLE_QWEN_IMAGE "Enable Qwen-Image model runner" ON)
option(DIFFUSION_SERVER_ENABLE_QWEN_IMAGE_CPU "Enable Qwen-Image CPU backend" ON)
option(DIFFUSION_SERVER_ENABLE_QWEN_IMAGE_HIP "Enable Qwen-Image HIP backend" ON)
option(DIFFUSION_SERVER_ENABLE_SAM3 "Enable SAM3 API surface" ON)
option(DIFFUSION_SERVER_ENABLE_MCP "Enable minimal MCP stdio mode" OFF)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

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
