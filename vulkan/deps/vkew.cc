// SPDX-License-Identifier: MIT
// Copyright 2025 - Present, Light Transport Entertainment Inc.
//
// vkew - Vulkan Extension Wrangler Implementation
//

#include "vkew.h"

#include <cstdio>
#include <cstring>

//------------------------------------------------------------------------------
// Platform-specific dynamic library loading
//------------------------------------------------------------------------------

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static HMODULE g_vkLibrary = nullptr;

static void* vkewLoadLibrary(void) {
    g_vkLibrary = LoadLibraryA("vulkan-1.dll");
    return g_vkLibrary;
}

static void vkewUnloadLibrary(void) {
    if (g_vkLibrary) {
        FreeLibrary(g_vkLibrary);
        g_vkLibrary = nullptr;
    }
}

static void* vkewGetProcAddress(const char* name) {
    if (!g_vkLibrary) return nullptr;
    return (void*)GetProcAddress(g_vkLibrary, name);
}

#elif defined(__APPLE__)
#include <dlfcn.h>

static void* g_vkLibrary = nullptr;

static void* vkewLoadLibrary(void) {
    // Try MoltenVK paths first, then standard Vulkan SDK
    const char* libPaths[] = {
        "libvulkan.1.dylib",
        "libvulkan.dylib",
        "/usr/local/lib/libvulkan.dylib",
        "/usr/local/lib/libvulkan.1.dylib",
        "libMoltenVK.dylib",
        nullptr
    };

    for (int i = 0; libPaths[i]; i++) {
        g_vkLibrary = dlopen(libPaths[i], RTLD_NOW | RTLD_LOCAL);
        if (g_vkLibrary) break;
    }
    return g_vkLibrary;
}

static void vkewUnloadLibrary(void) {
    if (g_vkLibrary) {
        dlclose(g_vkLibrary);
        g_vkLibrary = nullptr;
    }
}

static void* vkewGetProcAddress(const char* name) {
    if (!g_vkLibrary) return nullptr;
    return dlsym(g_vkLibrary, name);
}

#else // Linux / Unix
#include <dlfcn.h>

static void* g_vkLibrary = nullptr;

static void* vkewLoadLibrary(void) {
    // Try standard paths
    const char* libPaths[] = {
        "libvulkan.so.1",
        "libvulkan.so",
        "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
        "/usr/lib/libvulkan.so.1",
        "/usr/local/lib/libvulkan.so.1",
        nullptr
    };

    for (int i = 0; libPaths[i]; i++) {
        g_vkLibrary = dlopen(libPaths[i], RTLD_NOW | RTLD_LOCAL);
        if (g_vkLibrary) break;
    }
    return g_vkLibrary;
}

static void vkewUnloadLibrary(void) {
    if (g_vkLibrary) {
        dlclose(g_vkLibrary);
        g_vkLibrary = nullptr;
    }
}

static void* vkewGetProcAddress(const char* name) {
    if (!g_vkLibrary) return nullptr;
    return dlsym(g_vkLibrary, name);
}

#endif

//------------------------------------------------------------------------------
// Global state
//------------------------------------------------------------------------------

static bool g_initialized = false;
static char g_errorMessage[512] = "";

static void setError(const char* msg) {
    strncpy(g_errorMessage, msg, sizeof(g_errorMessage) - 1);
    g_errorMessage[sizeof(g_errorMessage) - 1] = '\0';
}

//------------------------------------------------------------------------------
// Function pointer definitions
//------------------------------------------------------------------------------

// Global/Loader functions
PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
PFN_vkCreateInstance vkCreateInstance = nullptr;

// Instance functions
PFN_vkDestroyInstance vkDestroyInstance = nullptr;
PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr;
PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = nullptr;
PFN_vkGetPhysicalDeviceFeatures vkGetPhysicalDeviceFeatures = nullptr;
PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = nullptr;
PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
PFN_vkCreateDevice vkCreateDevice = nullptr;
PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr = nullptr;

// Device functions
PFN_vkDestroyDevice vkDestroyDevice = nullptr;
PFN_vkGetDeviceQueue vkGetDeviceQueue = nullptr;
PFN_vkDeviceWaitIdle vkDeviceWaitIdle = nullptr;
PFN_vkQueueWaitIdle vkQueueWaitIdle = nullptr;

// Memory
PFN_vkAllocateMemory vkAllocateMemory = nullptr;
PFN_vkFreeMemory vkFreeMemory = nullptr;
PFN_vkMapMemory vkMapMemory = nullptr;
PFN_vkUnmapMemory vkUnmapMemory = nullptr;

// Buffer
PFN_vkCreateBuffer vkCreateBuffer = nullptr;
PFN_vkDestroyBuffer vkDestroyBuffer = nullptr;
PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = nullptr;
PFN_vkBindBufferMemory vkBindBufferMemory = nullptr;

// Shader
PFN_vkCreateShaderModule vkCreateShaderModule = nullptr;
PFN_vkDestroyShaderModule vkDestroyShaderModule = nullptr;

// Pipeline
PFN_vkCreateComputePipelines vkCreateComputePipelines = nullptr;
PFN_vkDestroyPipeline vkDestroyPipeline = nullptr;
PFN_vkCreatePipelineLayout vkCreatePipelineLayout = nullptr;
PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout = nullptr;

// Descriptor
PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout = nullptr;
PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout = nullptr;
PFN_vkCreateDescriptorPool vkCreateDescriptorPool = nullptr;
PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool = nullptr;
PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets = nullptr;
PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets = nullptr;

// Command pool/buffer
PFN_vkCreateCommandPool vkCreateCommandPool = nullptr;
PFN_vkDestroyCommandPool vkDestroyCommandPool = nullptr;
PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = nullptr;
PFN_vkFreeCommandBuffers vkFreeCommandBuffers = nullptr;
PFN_vkBeginCommandBuffer vkBeginCommandBuffer = nullptr;
PFN_vkEndCommandBuffer vkEndCommandBuffer = nullptr;

// Queue
PFN_vkQueueSubmit vkQueueSubmit = nullptr;

// Commands
PFN_vkCmdBindPipeline vkCmdBindPipeline = nullptr;
PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets = nullptr;
PFN_vkCmdPushConstants vkCmdPushConstants = nullptr;
PFN_vkCmdDispatch vkCmdDispatch = nullptr;
PFN_vkCmdCopyBuffer vkCmdCopyBuffer = nullptr;

//------------------------------------------------------------------------------
// Helper macro for loading functions
//------------------------------------------------------------------------------

#define VKEW_LOAD_GLOBAL(func) \
    func = (PFN_##func)vkewGetProcAddress(#func); \
    if (!func) { \
        setError("Failed to load " #func); \
        return false; \
    }

#define VKEW_LOAD_INSTANCE(func) \
    func = (PFN_##func)vkGetInstanceProcAddr(instance, #func); \
    if (!func) { \
        setError("Failed to load instance function " #func); \
        return false; \
    }

#define VKEW_LOAD_DEVICE(func) \
    func = (PFN_##func)vkGetDeviceProcAddr(device, #func); \
    if (!func) { \
        setError("Failed to load device function " #func); \
        return false; \
    }

//------------------------------------------------------------------------------
// VKEW API Implementation
//------------------------------------------------------------------------------

bool vkewInit(void) {
    if (g_initialized) {
        return true;
    }

    g_errorMessage[0] = '\0';

    // Load Vulkan library
    if (!vkewLoadLibrary()) {
        setError("Failed to load Vulkan library. Is Vulkan installed?");
        return false;
    }

    // Get vkGetInstanceProcAddr from the library
    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)vkewGetProcAddress("vkGetInstanceProcAddr");
    if (!vkGetInstanceProcAddr) {
        setError("Failed to get vkGetInstanceProcAddr");
        vkewUnloadLibrary();
        return false;
    }

    // Load global functions (can be loaded without an instance)
    vkCreateInstance = (PFN_vkCreateInstance)vkGetInstanceProcAddr(nullptr, "vkCreateInstance");
    if (!vkCreateInstance) {
        setError("Failed to load vkCreateInstance");
        vkewUnloadLibrary();
        return false;
    }

    g_initialized = true;
    return true;
}

void vkewShutdown(void) {
    if (!g_initialized) {
        return;
    }

    // Reset all function pointers
    vkGetInstanceProcAddr = nullptr;
    vkCreateInstance = nullptr;
    vkDestroyInstance = nullptr;
    vkEnumeratePhysicalDevices = nullptr;
    vkGetPhysicalDeviceProperties = nullptr;
    vkGetPhysicalDeviceFeatures = nullptr;
    vkGetPhysicalDeviceMemoryProperties = nullptr;
    vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
    vkCreateDevice = nullptr;
    vkGetDeviceProcAddr = nullptr;
    vkDestroyDevice = nullptr;
    vkGetDeviceQueue = nullptr;
    vkDeviceWaitIdle = nullptr;
    vkQueueWaitIdle = nullptr;
    vkAllocateMemory = nullptr;
    vkFreeMemory = nullptr;
    vkMapMemory = nullptr;
    vkUnmapMemory = nullptr;
    vkCreateBuffer = nullptr;
    vkDestroyBuffer = nullptr;
    vkGetBufferMemoryRequirements = nullptr;
    vkBindBufferMemory = nullptr;
    vkCreateShaderModule = nullptr;
    vkDestroyShaderModule = nullptr;
    vkCreateComputePipelines = nullptr;
    vkDestroyPipeline = nullptr;
    vkCreatePipelineLayout = nullptr;
    vkDestroyPipelineLayout = nullptr;
    vkCreateDescriptorSetLayout = nullptr;
    vkDestroyDescriptorSetLayout = nullptr;
    vkCreateDescriptorPool = nullptr;
    vkDestroyDescriptorPool = nullptr;
    vkAllocateDescriptorSets = nullptr;
    vkUpdateDescriptorSets = nullptr;
    vkCreateCommandPool = nullptr;
    vkDestroyCommandPool = nullptr;
    vkAllocateCommandBuffers = nullptr;
    vkFreeCommandBuffers = nullptr;
    vkBeginCommandBuffer = nullptr;
    vkEndCommandBuffer = nullptr;
    vkQueueSubmit = nullptr;
    vkCmdBindPipeline = nullptr;
    vkCmdBindDescriptorSets = nullptr;
    vkCmdPushConstants = nullptr;
    vkCmdDispatch = nullptr;
    vkCmdCopyBuffer = nullptr;

    vkewUnloadLibrary();
    g_initialized = false;
}

bool vkewIsInitialized(void) {
    return g_initialized;
}

bool vkewLoadInstance(VkInstance instance) {
    if (!g_initialized) {
        setError("vkew not initialized. Call vkewInit() first.");
        return false;
    }

    if (!instance) {
        setError("Invalid VkInstance (null)");
        return false;
    }

    // Load instance-level functions
    VKEW_LOAD_INSTANCE(vkDestroyInstance);
    VKEW_LOAD_INSTANCE(vkEnumeratePhysicalDevices);
    VKEW_LOAD_INSTANCE(vkGetPhysicalDeviceProperties);
    VKEW_LOAD_INSTANCE(vkGetPhysicalDeviceFeatures);
    VKEW_LOAD_INSTANCE(vkGetPhysicalDeviceMemoryProperties);
    VKEW_LOAD_INSTANCE(vkGetPhysicalDeviceQueueFamilyProperties);
    VKEW_LOAD_INSTANCE(vkCreateDevice);
    VKEW_LOAD_INSTANCE(vkGetDeviceProcAddr);

    return true;
}

bool vkewLoadDevice(VkDevice device) {
    if (!g_initialized) {
        setError("vkew not initialized. Call vkewInit() first.");
        return false;
    }

    if (!device) {
        setError("Invalid VkDevice (null)");
        return false;
    }

    // Load device-level functions
    VKEW_LOAD_DEVICE(vkDestroyDevice);
    VKEW_LOAD_DEVICE(vkGetDeviceQueue);
    VKEW_LOAD_DEVICE(vkDeviceWaitIdle);
    VKEW_LOAD_DEVICE(vkQueueWaitIdle);

    // Memory
    VKEW_LOAD_DEVICE(vkAllocateMemory);
    VKEW_LOAD_DEVICE(vkFreeMemory);
    VKEW_LOAD_DEVICE(vkMapMemory);
    VKEW_LOAD_DEVICE(vkUnmapMemory);

    // Buffer
    VKEW_LOAD_DEVICE(vkCreateBuffer);
    VKEW_LOAD_DEVICE(vkDestroyBuffer);
    VKEW_LOAD_DEVICE(vkGetBufferMemoryRequirements);
    VKEW_LOAD_DEVICE(vkBindBufferMemory);

    // Shader
    VKEW_LOAD_DEVICE(vkCreateShaderModule);
    VKEW_LOAD_DEVICE(vkDestroyShaderModule);

    // Pipeline
    VKEW_LOAD_DEVICE(vkCreateComputePipelines);
    VKEW_LOAD_DEVICE(vkDestroyPipeline);
    VKEW_LOAD_DEVICE(vkCreatePipelineLayout);
    VKEW_LOAD_DEVICE(vkDestroyPipelineLayout);

    // Descriptor
    VKEW_LOAD_DEVICE(vkCreateDescriptorSetLayout);
    VKEW_LOAD_DEVICE(vkDestroyDescriptorSetLayout);
    VKEW_LOAD_DEVICE(vkCreateDescriptorPool);
    VKEW_LOAD_DEVICE(vkDestroyDescriptorPool);
    VKEW_LOAD_DEVICE(vkAllocateDescriptorSets);
    VKEW_LOAD_DEVICE(vkUpdateDescriptorSets);

    // Command pool/buffer
    VKEW_LOAD_DEVICE(vkCreateCommandPool);
    VKEW_LOAD_DEVICE(vkDestroyCommandPool);
    VKEW_LOAD_DEVICE(vkAllocateCommandBuffers);
    VKEW_LOAD_DEVICE(vkFreeCommandBuffers);
    VKEW_LOAD_DEVICE(vkBeginCommandBuffer);
    VKEW_LOAD_DEVICE(vkEndCommandBuffer);

    // Queue
    VKEW_LOAD_DEVICE(vkQueueSubmit);

    // Commands
    VKEW_LOAD_DEVICE(vkCmdBindPipeline);
    VKEW_LOAD_DEVICE(vkCmdBindDescriptorSets);
    VKEW_LOAD_DEVICE(vkCmdPushConstants);
    VKEW_LOAD_DEVICE(vkCmdDispatch);
    VKEW_LOAD_DEVICE(vkCmdCopyBuffer);

    return true;
}

const char* vkewGetError(void) {
    return g_errorMessage;
}

const char* vkewResultToString(VkResult result) {
    switch (result) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_EVENT_SET: return "VK_EVENT_SET";
        case VK_EVENT_RESET: return "VK_EVENT_RESET";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
        case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        case VK_ERROR_FRAGMENTATION: return "VK_ERROR_FRAGMENTATION";
        case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS: return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
        case VK_PIPELINE_COMPILE_REQUIRED: return "VK_PIPELINE_COMPILE_REQUIRED";
        case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
        case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
        case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
        default: return "VK_UNKNOWN_RESULT";
    }
}
