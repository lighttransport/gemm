cmake_minimum_required(VERSION 3.16)
project(gemm_nn LANGUAGES C)
find_package(Threads REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gn_kernels.inc
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/embed_kernels.py
          ${CMAKE_CURRENT_SOURCE_DIR}/gn_kernels.cu ${CMAKE_CURRENT_BINARY_DIR}/gn_kernels.inc
  DEPENDS gn_kernels.cu embed_kernels.py VERBATIM)
add_library(gn STATIC gn.c gn_cpu.c gn_gpu.c ../cuda/cuew.c ../rdna4/rocew.c
  ${CMAKE_CURRENT_BINARY_DIR}/gn_kernels.inc)
set_target_properties(gn PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
target_include_directories(gn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(gn PUBLIC Threads::Threads m ${CMAKE_DL_LIBS})
target_compile_definitions(gn PRIVATE _POSIX_C_SOURCE=200809L)
add_executable(gn_tool gn_tool.c)
target_link_libraries(gn_tool PRIVATE gn)
if(BUILD_TESTING)
  add_executable(gn-test test_gn.c)
  target_link_libraries(gn-test PRIVATE gn)
  add_test(NAME gn-native COMMAND gn-test ${CMAKE_CURRENT_BINARY_DIR}/roundtrip.safetensors)
  add_executable(gn-gpu-test test_gpu.c)
  target_link_libraries(gn-gpu-test PRIVATE gn)
endif()
