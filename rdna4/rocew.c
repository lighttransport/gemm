/*
 * ROCEW - ROCm/HIP Extension Wrangler
 *
 * Copyright 2024 vision-language.cpp Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "rocew.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define VC_EXTRALEAN
#  include <windows.h>

typedef HMODULE DynamicLibrary;

#  define dynamic_library_open(path)        LoadLibraryA(path)
#  define dynamic_library_close(lib)        FreeLibrary(lib)
#  define dynamic_library_find(lib, symbol) GetProcAddress(lib, symbol)
#else
#  include <dlfcn.h>

typedef void* DynamicLibrary;

#  define dynamic_library_open(path)        dlopen(path, RTLD_NOW)
#  define dynamic_library_close(lib)        dlclose(lib)
#  define dynamic_library_find(lib, symbol) dlsym(lib, symbol)
#endif

#define HIP_LIBRARY_FIND(name) \
    name = (t##name)dynamic_library_find(hip_lib, #name)

#define HIPRTC_LIBRARY_FIND(name) \
    name = (t##name)dynamic_library_find(hiprtc_lib, #name)

/* Library handles */
static DynamicLibrary hip_lib = NULL;
static DynamicLibrary hiprtc_lib = NULL;
static int hip_available = 0;
static int hiprtc_available = 0;

/* ============================================================================
 * HIP Function Definitions
 * ============================================================================ */

/* Error handling */
thipGetErrorName hipGetErrorName = NULL;
thipGetErrorString hipGetErrorString = NULL;
thipGetLastError hipGetLastError = NULL;
thipPeekAtLastError hipPeekAtLastError = NULL;

/* Initialization */
thipInit hipInit = NULL;

/* Device management */
thipGetDeviceCount hipGetDeviceCount = NULL;
thipGetDevice hipGetDevice = NULL;
thipSetDevice hipSetDevice = NULL;
thipGetDeviceProperties hipGetDeviceProperties = NULL;
thipDeviceGetAttribute hipDeviceGetAttribute = NULL;
thipDeviceGetName hipDeviceGetName = NULL;
thipDeviceGetPCIBusId hipDeviceGetPCIBusId = NULL;
thipDeviceGetByPCIBusId hipDeviceGetByPCIBusId = NULL;
thipDeviceTotalMem hipDeviceTotalMem = NULL;
thipDeviceSynchronize hipDeviceSynchronize = NULL;
thipDeviceReset hipDeviceReset = NULL;

/* Context management */
thipCtxCreate hipCtxCreate = NULL;
thipCtxDestroy hipCtxDestroy = NULL;
thipCtxPushCurrent hipCtxPushCurrent = NULL;
thipCtxPopCurrent hipCtxPopCurrent = NULL;
thipCtxSetCurrent hipCtxSetCurrent = NULL;
thipCtxGetCurrent hipCtxGetCurrent = NULL;
thipCtxGetDevice hipCtxGetDevice = NULL;
thipCtxSynchronize hipCtxSynchronize = NULL;

/* Memory management */
thipMalloc hipMalloc = NULL;
thipMallocPitch hipMallocPitch = NULL;
thipMalloc3D hipMalloc3D = NULL;
thipFree hipFree = NULL;
thipMallocHost hipMallocHost = NULL;
thipHostMalloc hipHostMalloc = NULL;
thipFreeHost hipFreeHost = NULL;
thipHostFree hipHostFree = NULL;
thipMallocManaged hipMallocManaged = NULL;
thipMemcpy hipMemcpy = NULL;
thipMemcpyAsync hipMemcpyAsync = NULL;
thipMemcpyHtoD hipMemcpyHtoD = NULL;
thipMemcpyDtoH hipMemcpyDtoH = NULL;
thipMemcpyDtoD hipMemcpyDtoD = NULL;
thipMemcpyHtoDAsync hipMemcpyHtoDAsync = NULL;
thipMemcpyDtoHAsync hipMemcpyDtoHAsync = NULL;
thipMemcpyDtoDAsync hipMemcpyDtoDAsync = NULL;
thipMemset hipMemset = NULL;
thipMemsetAsync hipMemsetAsync = NULL;
thipMemsetD8 hipMemsetD8 = NULL;
thipMemsetD16 hipMemsetD16 = NULL;
thipMemsetD32 hipMemsetD32 = NULL;
thipMemGetInfo hipMemGetInfo = NULL;
thipMemPtrGetInfo hipMemPtrGetInfo = NULL;

/* Module management */
thipModuleLoad hipModuleLoad = NULL;
thipModuleLoadData hipModuleLoadData = NULL;
thipModuleLoadDataEx hipModuleLoadDataEx = NULL;
thipModuleUnload hipModuleUnload = NULL;
thipModuleGetFunction hipModuleGetFunction = NULL;
thipModuleGetGlobal hipModuleGetGlobal = NULL;

/* Function management */
thipFuncGetAttribute hipFuncGetAttribute = NULL;
thipFuncSetAttribute hipFuncSetAttribute = NULL;
thipFuncSetCacheConfig hipFuncSetCacheConfig = NULL;
thipFuncSetSharedMemConfig hipFuncSetSharedMemConfig = NULL;

/* Kernel launch */
thipModuleLaunchKernel hipModuleLaunchKernel = NULL;

/* Stream management */
thipStreamCreate hipStreamCreate = NULL;
thipStreamCreateWithFlags hipStreamCreateWithFlags = NULL;
thipStreamCreateWithPriority hipStreamCreateWithPriority = NULL;
thipStreamDestroy hipStreamDestroy = NULL;
thipStreamQuery hipStreamQuery = NULL;
thipStreamSynchronize hipStreamSynchronize = NULL;
thipStreamWaitEvent hipStreamWaitEvent = NULL;
thipStreamGetFlags hipStreamGetFlags = NULL;
thipStreamGetPriority hipStreamGetPriority = NULL;

/* Event management */
thipEventCreate hipEventCreate = NULL;
thipEventCreateWithFlags hipEventCreateWithFlags = NULL;
thipEventDestroy hipEventDestroy = NULL;
thipEventRecord hipEventRecord = NULL;
thipEventQuery hipEventQuery = NULL;
thipEventSynchronize hipEventSynchronize = NULL;
thipEventElapsedTime hipEventElapsedTime = NULL;

/* Occupancy */
thipOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor = NULL;
thipOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize = NULL;

/* Graph capture */
thipStreamBeginCapture hipStreamBeginCapture = NULL;
thipStreamEndCapture   hipStreamEndCapture   = NULL;
thipGraphInstantiate   hipGraphInstantiate   = NULL;
thipGraphLaunch        hipGraphLaunch        = NULL;
thipGraphExecDestroy   hipGraphExecDestroy   = NULL;
thipGraphDestroy       hipGraphDestroy       = NULL;

/* ============================================================================
 * HIPRTC Function Definitions
 * ============================================================================ */

thiprtcGetErrorString hiprtcGetErrorString = NULL;
thiprtcVersion hiprtcVersion = NULL;
thiprtcCreateProgram hiprtcCreateProgram = NULL;
thiprtcDestroyProgram hiprtcDestroyProgram = NULL;
thiprtcCompileProgram hiprtcCompileProgram = NULL;
thiprtcGetProgramLogSize hiprtcGetProgramLogSize = NULL;
thiprtcGetProgramLog hiprtcGetProgramLog = NULL;
thiprtcGetCodeSize hiprtcGetCodeSize = NULL;
thiprtcGetCode hiprtcGetCode = NULL;
thiprtcGetBitcodeSize hiprtcGetBitcodeSize = NULL;
thiprtcGetBitcode hiprtcGetBitcode = NULL;
thiprtcAddNameExpression hiprtcAddNameExpression = NULL;
thiprtcGetLoweredName hiprtcGetLoweredName = NULL;

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

static DynamicLibrary dynamic_library_open_find(const char** paths)
{
    int i = 0;
    while (paths[i] != NULL) {
        DynamicLibrary lib = dynamic_library_open(paths[i]);
        if (lib != NULL) {
            return lib;
        }
        ++i;
    }
    return NULL;
}

static void rocewExit(void)
{
    if (hip_lib != NULL) {
        dynamic_library_close(hip_lib);
        hip_lib = NULL;
    }
    if (hiprtc_lib != NULL) {
        dynamic_library_close(hiprtc_lib);
        hiprtc_lib = NULL;
    }
    hip_available = 0;
    hiprtc_available = 0;
}

static int loadHIP(void)
{
#ifdef _WIN32
    const char* hip_paths[] = {
        "amdhip64_6.dll",
        "amdhip64.dll",
        "amdhip64_5.dll",
        "C:\\Program Files\\AMD\\ROCm\\6.4\\bin\\amdhip64_6.dll",
        "C:\\Program Files\\AMD\\ROCm\\5.5\\bin\\amdhip64.dll",
        NULL
    };
#elif defined(__APPLE__)
    /* HIP is not supported on macOS */
    const char* hip_paths[] = { NULL };
#else
    const char* hip_paths[] = {
        "libamdhip64.so",
        "libamdhip64.so.6",
        "libamdhip64.so.5",
        "/opt/rocm/lib/libamdhip64.so",
        "/opt/rocm/lib/libamdhip64.so.6",
        "/opt/rocm/lib/libamdhip64.so.5",
        NULL
    };
#endif

    hip_lib = dynamic_library_open_find(hip_paths);
    if (hip_lib == NULL) {
        return ROCEW_ERROR_OPEN_FAILED;
    }

    /* Load HIP functions */

    /* Error handling */
    HIP_LIBRARY_FIND(hipGetErrorName);
    HIP_LIBRARY_FIND(hipGetErrorString);
    HIP_LIBRARY_FIND(hipGetLastError);
    HIP_LIBRARY_FIND(hipPeekAtLastError);

    /* Initialization */
    HIP_LIBRARY_FIND(hipInit);

    /* Device management */
    HIP_LIBRARY_FIND(hipGetDeviceCount);
    HIP_LIBRARY_FIND(hipGetDevice);
    HIP_LIBRARY_FIND(hipSetDevice);
    HIP_LIBRARY_FIND(hipGetDeviceProperties);
    HIP_LIBRARY_FIND(hipDeviceGetAttribute);
    HIP_LIBRARY_FIND(hipDeviceGetName);
    HIP_LIBRARY_FIND(hipDeviceGetPCIBusId);
    HIP_LIBRARY_FIND(hipDeviceGetByPCIBusId);
    HIP_LIBRARY_FIND(hipDeviceTotalMem);
    HIP_LIBRARY_FIND(hipDeviceSynchronize);
    HIP_LIBRARY_FIND(hipDeviceReset);

    /* Context management */
    HIP_LIBRARY_FIND(hipCtxCreate);
    HIP_LIBRARY_FIND(hipCtxDestroy);
    HIP_LIBRARY_FIND(hipCtxPushCurrent);
    HIP_LIBRARY_FIND(hipCtxPopCurrent);
    HIP_LIBRARY_FIND(hipCtxSetCurrent);
    HIP_LIBRARY_FIND(hipCtxGetCurrent);
    HIP_LIBRARY_FIND(hipCtxGetDevice);
    HIP_LIBRARY_FIND(hipCtxSynchronize);

    /* Memory management */
    HIP_LIBRARY_FIND(hipMalloc);
    HIP_LIBRARY_FIND(hipMallocPitch);
    HIP_LIBRARY_FIND(hipMalloc3D);
    HIP_LIBRARY_FIND(hipFree);
    HIP_LIBRARY_FIND(hipMallocHost);
    HIP_LIBRARY_FIND(hipHostMalloc);
    HIP_LIBRARY_FIND(hipFreeHost);
    HIP_LIBRARY_FIND(hipHostFree);
    HIP_LIBRARY_FIND(hipMallocManaged);
    HIP_LIBRARY_FIND(hipMemcpy);
    HIP_LIBRARY_FIND(hipMemcpyAsync);
    HIP_LIBRARY_FIND(hipMemcpyHtoD);
    HIP_LIBRARY_FIND(hipMemcpyDtoH);
    HIP_LIBRARY_FIND(hipMemcpyDtoD);
    HIP_LIBRARY_FIND(hipMemcpyHtoDAsync);
    HIP_LIBRARY_FIND(hipMemcpyDtoHAsync);
    HIP_LIBRARY_FIND(hipMemcpyDtoDAsync);
    HIP_LIBRARY_FIND(hipMemset);
    HIP_LIBRARY_FIND(hipMemsetAsync);
    HIP_LIBRARY_FIND(hipMemsetD8);
    HIP_LIBRARY_FIND(hipMemsetD16);
    HIP_LIBRARY_FIND(hipMemsetD32);
    HIP_LIBRARY_FIND(hipMemGetInfo);
    HIP_LIBRARY_FIND(hipMemPtrGetInfo);

    /* Module management */
    HIP_LIBRARY_FIND(hipModuleLoad);
    HIP_LIBRARY_FIND(hipModuleLoadData);
    HIP_LIBRARY_FIND(hipModuleLoadDataEx);
    HIP_LIBRARY_FIND(hipModuleUnload);
    HIP_LIBRARY_FIND(hipModuleGetFunction);
    HIP_LIBRARY_FIND(hipModuleGetGlobal);

    /* Function management */
    HIP_LIBRARY_FIND(hipFuncGetAttribute);
    HIP_LIBRARY_FIND(hipFuncSetAttribute);
    HIP_LIBRARY_FIND(hipFuncSetCacheConfig);
    HIP_LIBRARY_FIND(hipFuncSetSharedMemConfig);

    /* Kernel launch */
    HIP_LIBRARY_FIND(hipModuleLaunchKernel);

    /* Stream management */
    HIP_LIBRARY_FIND(hipStreamCreate);
    HIP_LIBRARY_FIND(hipStreamCreateWithFlags);
    HIP_LIBRARY_FIND(hipStreamCreateWithPriority);
    HIP_LIBRARY_FIND(hipStreamDestroy);
    HIP_LIBRARY_FIND(hipStreamQuery);
    HIP_LIBRARY_FIND(hipStreamSynchronize);
    HIP_LIBRARY_FIND(hipStreamWaitEvent);
    HIP_LIBRARY_FIND(hipStreamGetFlags);
    HIP_LIBRARY_FIND(hipStreamGetPriority);

    /* Event management */
    HIP_LIBRARY_FIND(hipEventCreate);
    HIP_LIBRARY_FIND(hipEventCreateWithFlags);
    HIP_LIBRARY_FIND(hipEventDestroy);
    HIP_LIBRARY_FIND(hipEventRecord);
    HIP_LIBRARY_FIND(hipEventQuery);
    HIP_LIBRARY_FIND(hipEventSynchronize);
    HIP_LIBRARY_FIND(hipEventElapsedTime);

    /* Occupancy */
    HIP_LIBRARY_FIND(hipOccupancyMaxActiveBlocksPerMultiprocessor);
    HIP_LIBRARY_FIND(hipOccupancyMaxPotentialBlockSize);

    /* Graph capture (optional — older ROCm may not export these) */
    HIP_LIBRARY_FIND(hipStreamBeginCapture);
    HIP_LIBRARY_FIND(hipStreamEndCapture);
    HIP_LIBRARY_FIND(hipGraphInstantiate);
    HIP_LIBRARY_FIND(hipGraphLaunch);
    HIP_LIBRARY_FIND(hipGraphExecDestroy);
    HIP_LIBRARY_FIND(hipGraphDestroy);

    hip_available = 1;
    return ROCEW_SUCCESS;
}

static int loadHIPRTC(void)
{
#ifdef _WIN32
    const char* hiprtc_paths[] = {
        "hiprtc0604.dll",
        "hiprtc.dll",
        "hiprtc0601.dll",
        "hiprtc0600.dll",
        "hiprtc0506.dll",
        "C:\\Program Files\\AMD\\ROCm\\6.4\\bin\\hiprtc0604.dll",
        NULL
    };
#elif defined(__APPLE__)
    /* HIPRTC is not supported on macOS */
    const char* hiprtc_paths[] = { NULL };
#else
    const char* hiprtc_paths[] = {
        "libhiprtc.so",
        "libhiprtc.so.6",
        "libhiprtc.so.5",
        "/opt/rocm/lib/libhiprtc.so",
        "/opt/rocm/lib/libhiprtc.so.6",
        "/opt/rocm/lib/libhiprtc.so.5",
        NULL
    };
#endif

    hiprtc_lib = dynamic_library_open_find(hiprtc_paths);
    if (hiprtc_lib == NULL) {
        return ROCEW_ERROR_HIPRTC_OPEN_FAILED;
    }

    /* Load HIPRTC functions */
    HIPRTC_LIBRARY_FIND(hiprtcGetErrorString);
    HIPRTC_LIBRARY_FIND(hiprtcVersion);
    HIPRTC_LIBRARY_FIND(hiprtcCreateProgram);
    HIPRTC_LIBRARY_FIND(hiprtcDestroyProgram);
    HIPRTC_LIBRARY_FIND(hiprtcCompileProgram);
    HIPRTC_LIBRARY_FIND(hiprtcGetProgramLogSize);
    HIPRTC_LIBRARY_FIND(hiprtcGetProgramLog);
    HIPRTC_LIBRARY_FIND(hiprtcGetCodeSize);
    HIPRTC_LIBRARY_FIND(hiprtcGetCode);
    HIPRTC_LIBRARY_FIND(hiprtcGetBitcodeSize);
    HIPRTC_LIBRARY_FIND(hiprtcGetBitcode);
    HIPRTC_LIBRARY_FIND(hiprtcAddNameExpression);
    HIPRTC_LIBRARY_FIND(hiprtcGetLoweredName);

    hiprtc_available = 1;
    return ROCEW_SUCCESS;
}

/* ============================================================================
 * Public API
 * ============================================================================ */

int rocewInit(unsigned int flags)
{
    int error;
    static int atexit_registered = 0;

    /* Register cleanup at exit (only once) */
    if (!atexit_registered) {
        error = atexit(rocewExit);
        if (error) {
            return ROCEW_ERROR_ATEXIT_FAILED;
        }
        atexit_registered = 1;
    }

    /* Load HIP if requested and not already loaded */
    if ((flags & ROCEW_INIT_HIP) && !hip_available) {
        error = loadHIP();
        if (error != ROCEW_SUCCESS) {
            return error;
        }
    }

    /* Load HIPRTC if requested and not already loaded */
    if ((flags & ROCEW_INIT_HIPRTC) && !hiprtc_available) {
        error = loadHIPRTC();
        if (error != ROCEW_SUCCESS) {
            /* HIPRTC failure is not fatal if HIP loaded successfully */
            if (!(flags & ROCEW_INIT_HIP) || !hip_available) {
                return error;
            }
        }
    }

    return ROCEW_SUCCESS;
}

int rocewHipAvailable(void)
{
    return hip_available;
}

int rocewHiprtcAvailable(void)
{
    return hiprtc_available;
}

const char* rocewGetRDNA4ArchString(int deviceId)
{
    static char archName[256] = {0};

    if (!hip_available || !hipGetDeviceProperties) {
        return NULL;
    }

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, deviceId) != hipSuccess) {
        return NULL;
    }

    /* RDNA4 architectures:
     * gfx1200 - Navi 48 (RX 9070 series)
     * gfx1201 - Navi 44 (lower-end RDNA4)
     */
    strncpy(archName, props.gcnArchName, sizeof(archName) - 1);
    archName[sizeof(archName) - 1] = '\0';

    return archName;
}

int rocewIsRDNA4(int deviceId)
{
    const char* arch = rocewGetRDNA4ArchString(deviceId);
    if (arch == NULL) {
        return 0;
    }

    /* Check for RDNA4 architecture identifiers */
    if (strncmp(arch, "gfx1200", 7) == 0 ||
        strncmp(arch, "gfx1201", 7) == 0 ||
        strncmp(arch, "gfx12", 5) == 0) {
        return 1;
    }

    return 0;
}
