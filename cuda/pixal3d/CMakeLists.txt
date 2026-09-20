cmake_minimum_required(VERSION 3.24)
project(pixal3d_cuda LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING
        "CUDA architectures for Pixal3D (for example 86 for RTX 3070)")
endif()

get_filename_component(PIXAL3D_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(FA2_INCLUDE "${PIXAL3D_ROOT}/tmp/pixal3d/fa2.inc")
add_custom_command(
    OUTPUT "${FA2_INCLUDE}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${PIXAL3D_ROOT}/tmp/pixal3d"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/extract_fa2.py"
            "${FA2_INCLUDE}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/extract_fa2.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/../fa2/cuda_fa2_kernels.h"
    VERBATIM)

add_library(pixal3d_cuda SHARED gpu.cu "${FA2_INCLUDE}")
set_target_properties(pixal3d_cuda PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    WINDOWS_EXPORT_ALL_SYMBOLS YES)
target_compile_options(pixal3d_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
if(MSVC)
    target_compile_options(pixal3d_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()
target_link_libraries(pixal3d_cuda PRIVATE CUDA::cublas CUDA::cudart)
