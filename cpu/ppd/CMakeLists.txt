cmake_minimum_required(VERSION 3.10)
project(cpu_ppd C)

set(COMMON ${CMAKE_CURRENT_SOURCE_DIR}/../../common)

get_filename_component(_CC_NAME "${CMAKE_C_COMPILER}" NAME)
if(_CC_NAME MATCHES "^fcc" OR _CC_NAME MATCHES "^FCC")
    set(FUJITSU_COMPILER TRUE)
else()
    set(FUJITSU_COMPILER FALSE)
endif()

set(COMMON_FLAGS -Wall -Wextra -Wno-unused-function -Wno-unused-parameter)
if(FUJITSU_COMPILER)
    list(APPEND COMMON_FLAGS -Nclang)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    set(ARCH_FLAGS -march=armv8.2-a+sve -ffp-contract=fast)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(ARCH_FLAGS -mavx2 -mfma -mf16c)
endif()

add_executable(test_ppd test_ppd.c)
target_include_directories(test_ppd PRIVATE ${COMMON})
target_compile_options(test_ppd PRIVATE ${COMMON_FLAGS} ${ARCH_FLAGS})
target_link_libraries(test_ppd PRIVATE m pthread)
