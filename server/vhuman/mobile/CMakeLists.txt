cmake_minimum_required(VERSION 3.20)
project(vhuman_mobile LANGUAGES CXX)
add_library(vhuman_mobile STATIC native.cpp)
target_compile_features(vhuman_mobile PUBLIC cxx_std_17)
target_include_directories(vhuman_mobile PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(vhuman_mobile PRIVATE -Wall -Wextra -Wpedantic)
set(FILAMENT_ROOT "" CACHE PATH "Extracted Filament 1.77.2 SDK")
set(FILAMENT_LIB_DIR "${FILAMENT_ROOT}/lib/x86_64" CACHE PATH "SDK static library directory")
if(FILAMENT_ROOT)
    add_library(vhuman_player STATIC player.cpp)
    target_compile_features(vhuman_player PUBLIC cxx_std_20)
    target_include_directories(vhuman_player SYSTEM PRIVATE "${FILAMENT_ROOT}/include")
    target_include_directories(vhuman_player PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(vhuman_player PRIVATE -Wall -Wextra -Wpedantic)
    set(filament_libs gltfio gltfio_core filamat filament backend filabridge filaflat
        utils geometry dracodec stb uberarchive uberzlib zstd mikktspace meshoptimizer shaders smol-v)
    if(NOT APPLE)
        list(APPEND filament_libs bluevk bluegl)
        target_link_libraries(vhuman_player PUBLIC "-Wl,--start-group")
    endif()
    foreach(lib IN LISTS filament_libs)
        if(NOT EXISTS "${FILAMENT_LIB_DIR}/lib${lib}.a")
            message(FATAL_ERROR "Missing SDK library: ${FILAMENT_LIB_DIR}/lib${lib}.a")
        endif()
        target_link_libraries(vhuman_player PUBLIC "${FILAMENT_LIB_DIR}/lib${lib}.a")
    endforeach()
    if(APPLE)
        target_link_libraries(vhuman_player PUBLIC "-framework Metal" "-framework QuartzCore"
            "-framework Foundation" "-framework CoreVideo" "-framework CoreGraphics" "-framework Accelerate")
    else()
        target_link_libraries(vhuman_player PUBLIC "-Wl,--end-group" pthread dl "-Wl,-z,noexecstack")
    endif()
    target_link_libraries(vhuman_player PUBLIC vhuman_mobile)
    if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        enable_language(Swift OBJCXX)
        add_executable(VHuman MACOSX_BUNDLE ios/App.swift ios/AvatarPackage.swift
            ios/AudioMotion.swift ios/Renderer.mm)
        target_link_libraries(VHuman PRIVATE vhuman_player "-framework UIKit" "-framework AVFoundation" "-framework CryptoKit")
        set_target_properties(VHuman PROPERTIES
            OBJCXX_STANDARD 20
            MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/ios/Info.plist"
            XCODE_ATTRIBUTE_SWIFT_VERSION "5.0"
            XCODE_ATTRIBUTE_SWIFT_OBJC_BRIDGING_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/ios/Renderer.h"
            XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
            XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "org.pixal3d.vhuman"
            XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2")
    else()
        add_executable(vhuman_player_probe player_probe.cpp)
        target_link_libraries(vhuman_player_probe PRIVATE vhuman_player)
    endif()
endif()
