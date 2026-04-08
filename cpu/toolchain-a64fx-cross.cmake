# CMake toolchain file for A64FX cross-compilation with Fujitsu compilers
#
# Usage:
#   mkdir build_a64fx && cd build_a64fx
#   cmake -DCMAKE_TOOLCHAIN_FILE=../toolchain-a64fx-cross.cmake ..
#   make

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER fccpx)
set(CMAKE_CXX_COMPILER FCCpx)

set(CMAKE_C_FLAGS_INIT "-Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast")
set(CMAKE_CXX_FLAGS_INIT "-Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
