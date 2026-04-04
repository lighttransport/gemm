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
 *
 * Dynamic loading library for AMD ROCm/HIP without requiring ROCm SDK at compile time.
 * Supports runtime kernel compilation via hiprtc.
 * Focused on RDNA4 architecture support.
 */

#ifndef ROCEW_H_
#define ROCEW_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/* Version */
#define ROCEW_VERSION_MAJOR 1
#define ROCEW_VERSION_MINOR 0

/* Error codes for rocewInit */
#define ROCEW_SUCCESS 0
#define ROCEW_ERROR_OPEN_FAILED -1
#define ROCEW_ERROR_ATEXIT_FAILED -2
#define ROCEW_ERROR_HIPRTC_OPEN_FAILED -3

/* Initialization flags */
#define ROCEW_INIT_HIP     (1 << 0)
#define ROCEW_INIT_HIPRTC  (1 << 1)
#define ROCEW_INIT_HIPBLAS (1 << 2)

/* HIP calling convention */
#ifdef _WIN32
/* ROCm/HIP on Windows uses __cdecl (default C calling convention) */
#  define HIPAPI
#else
#  define HIPAPI
#endif

/* ============================================================================
 * HIP Types
 * ============================================================================ */

typedef int hipError_t;
typedef int hipDevice_t;
typedef struct ihipCtx_t* hipCtx_t;
typedef struct ihipStream_t* hipStream_t;
typedef struct ihipModule_t* hipModule_t;
typedef struct ihipModuleSymbol_t* hipFunction_t;
typedef struct ihipEvent_t* hipEvent_t;
typedef void* hipDeviceptr_t;

/* hipDeviceArch_t - architecture flags (not used in ROCm 7.x props query) */
typedef struct {
    unsigned int hasGlobalInt32Atomics : 1;
    unsigned int hasGlobalFloatAtomicExch : 1;
    unsigned int hasSharedInt32Atomics : 1;
    unsigned int hasSharedFloatAtomicExch : 1;
    unsigned int hasFloatAtomicAdd : 1;
    unsigned int hasGlobalInt64Atomics : 1;
    unsigned int hasSharedInt64Atomics : 1;
    unsigned int hasDoubles : 1;
    unsigned int hasWarpVote : 1;
    unsigned int hasWarpBallot : 1;
    unsigned int hasWarpShuffle : 1;
    unsigned int hasFunnelShift : 1;
    unsigned int hasThreadFenceSystem : 1;
    unsigned int hasSyncThreadsExt : 1;
    unsigned int hasSurfaceFuncs : 1;
    unsigned int has3dGrid : 1;
    unsigned int hasDynamicParallelism : 1;
} hipDeviceArch_t;

/* hipDeviceProp_t - ROCm 6.4+ ABI compatible structure
 * The actual ROCm hipDeviceProp_t is very large (~1200+ bytes)
 * We define a large enough buffer to avoid stack overflow when
 * hipGetDeviceProperties writes to it.
 *
 * Key fields at known offsets:
 *   offset 0:   name[256]
 *   offset 256: totalGlobalMem (size_t)
 *   offset 264: sharedMemPerBlock (size_t)
 *   ...
 *   offset 328: major (int)
 *   offset 332: minor (int)
 *   offset 336: multiProcessorCount (int)
 *   ...
 *   offset 396: gcnArchName[256]
 */
typedef struct {
    char name[256];                     /* offset 0 */
    size_t totalGlobalMem;              /* offset 256 */
    size_t sharedMemPerBlock;           /* offset 264 */
    int regsPerBlock;                   /* offset 272 */
    int warpSize;                       /* offset 276 */
    size_t memPitch;                    /* offset 280 */
    int maxThreadsPerBlock;             /* offset 288 */
    int maxThreadsDim[3];               /* offset 292 */
    int maxGridSize[3];                 /* offset 304 */
    int clockRate;                      /* offset 316 */
    size_t totalConstMem;               /* offset 320 */
    int major;                          /* offset 328 */
    int minor;                          /* offset 332 */
    int multiProcessorCount;            /* offset 336 */
    int l2CacheSize;                    /* offset 340 */
    int maxThreadsPerMultiProcessor;    /* offset 344 */
    int computeMode;                    /* offset 348 */
    int clockInstructionRate;           /* offset 352 */
    int _reserved1[10];                 /* offset 356-395 (padding) */
    char gcnArchName[256];              /* offset 396 */
    /* Large padding to accommodate full ROCm 6.4+ structure */
    char _reserved2[512];               /* padding to ~1200 bytes total */
} hipDeviceProp_t;

typedef enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4
} hipMemcpyKind;

typedef enum hipDeviceAttribute_t {
    hipDeviceAttributeMaxThreadsPerBlock = 0,
    hipDeviceAttributeMaxBlockDimX = 1,
    hipDeviceAttributeMaxBlockDimY = 2,
    hipDeviceAttributeMaxBlockDimZ = 3,
    hipDeviceAttributeMaxGridDimX = 4,
    hipDeviceAttributeMaxGridDimY = 5,
    hipDeviceAttributeMaxGridDimZ = 6,
    hipDeviceAttributeMaxSharedMemoryPerBlock = 7,
    hipDeviceAttributeTotalConstantMemory = 8,
    hipDeviceAttributeWarpSize = 9,
    hipDeviceAttributeMaxRegistersPerBlock = 10,
    hipDeviceAttributeClockRate = 11,
    hipDeviceAttributeMemoryClockRate = 12,
    hipDeviceAttributeGlobalMemoryBusWidth = 13,
    hipDeviceAttributeMultiprocessorCount = 14,
    hipDeviceAttributeComputeCapabilityMajor = 15,
    hipDeviceAttributeComputeCapabilityMinor = 16,
    hipDeviceAttributeL2CacheSize = 17,
    hipDeviceAttributeMaxThreadsPerMultiProcessor = 18,
    hipDeviceAttributeComputeMode = 19,
    hipDeviceAttributeConcurrentKernels = 20,
    hipDeviceAttributePciBusId = 21,
    hipDeviceAttributePciDeviceId = 22,
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor = 23,
    hipDeviceAttributeIsMultiGpuBoard = 24,
    hipDeviceAttributeIntegrated = 25,
    hipDeviceAttributeCooperativeLaunch = 26,
    hipDeviceAttributeCooperativeMultiDeviceLaunch = 27,
    hipDeviceAttributeMaxTexture1DWidth = 28,
    hipDeviceAttributeMaxTexture2DWidth = 29,
    hipDeviceAttributeMaxTexture2DHeight = 30,
    hipDeviceAttributeMaxTexture3DWidth = 31,
    hipDeviceAttributeMaxTexture3DHeight = 32,
    hipDeviceAttributeMaxTexture3DDepth = 33,
    hipDeviceAttributeMaxPitch = 34,
    hipDeviceAttributeTextureAlignment = 35,
    hipDeviceAttributeKernelExecTimeout = 36,
    hipDeviceAttributeCanMapHostMemory = 37,
    hipDeviceAttributeEccEnabled = 38,
    hipDeviceAttributeGcnArch = 39,
    hipDeviceAttributeGcnArchName = 40,
    hipDeviceAttributePageableMemoryAccess = 41,
    hipDeviceAttributePageableMemoryAccessUsesHostPageTables = 42,
    hipDeviceAttributeDirectManagedMemAccessFromHost = 43,
    hipDeviceAttributeAsicRevision = 44
} hipDeviceAttribute_t;

typedef enum hipFuncAttribute_t {
    hipFuncAttributeMaxDynamicSharedMemorySize = 0,
    hipFuncAttributePreferredSharedMemoryCarveout = 1
} hipFuncAttribute_t;

/* HIP error codes */
#define hipSuccess                           0
#define hipErrorInvalidValue                 1
#define hipErrorOutOfMemory                  2
#define hipErrorMemoryAllocation             2
#define hipErrorNotInitialized               3
#define hipErrorInitializationError          3
#define hipErrorDeinitialized                4
#define hipErrorProfilerDisabled             5
#define hipErrorProfilerNotInitialized       6
#define hipErrorProfilerAlreadyStarted       7
#define hipErrorProfilerAlreadyStopped       8
#define hipErrorInvalidConfiguration         9
#define hipErrorInvalidPitchValue           12
#define hipErrorInvalidSymbol               13
#define hipErrorInvalidDevicePointer        17
#define hipErrorInvalidMemcpyDirection      21
#define hipErrorInsufficientDriver          35
#define hipErrorMissingConfiguration        52
#define hipErrorPriorLaunchFailure          53
#define hipErrorInvalidDeviceFunction       98
#define hipErrorNoDevice                   100
#define hipErrorInvalidDevice              101
#define hipErrorInvalidImage               200
#define hipErrorInvalidContext             201
#define hipErrorContextAlreadyCurrent      202
#define hipErrorMapFailed                  205
#define hipErrorUnmapFailed                206
#define hipErrorArrayIsMapped              207
#define hipErrorAlreadyMapped              208
#define hipErrorNoBinaryForGpu             209
#define hipErrorAlreadyAcquired            210
#define hipErrorNotMapped                  211
#define hipErrorNotMappedAsArray           212
#define hipErrorNotMappedAsPointer         213
#define hipErrorECCNotCorrectable          214
#define hipErrorUnsupportedLimit           215
#define hipErrorContextAlreadyInUse        216
#define hipErrorPeerAccessUnsupported      217
#define hipErrorInvalidKernelFile          218
#define hipErrorInvalidGraphicsContext     219
#define hipErrorInvalidSource              300
#define hipErrorFileNotFound               301
#define hipErrorSharedObjectSymbolNotFound 302
#define hipErrorSharedObjectInitFailed     303
#define hipErrorOperatingSystem            304
#define hipErrorInvalidHandle              400
#define hipErrorInvalidResourceHandle      400
#define hipErrorIllegalState               401
#define hipErrorNotFound                   500
#define hipErrorNotReady                   600
#define hipErrorIllegalAddress             700
#define hipErrorLaunchOutOfResources       701
#define hipErrorLaunchTimeOut              702
#define hipErrorPeerAccessAlreadyEnabled   704
#define hipErrorPeerAccessNotEnabled       705
#define hipErrorSetOnActiveProcess         708
#define hipErrorContextIsDestroyed         709
#define hipErrorAssert                     710
#define hipErrorHostMemoryAlreadyRegistered 712
#define hipErrorHostMemoryNotRegistered    713
#define hipErrorLaunchFailure              719
#define hipErrorCooperativeLaunchTooLarge  720
#define hipErrorNotSupported               801
#define hipErrorStreamCaptureUnsupported   900
#define hipErrorStreamCaptureInvalidated   901
#define hipErrorStreamCaptureMerge         902
#define hipErrorStreamCaptureUnmatched     903
#define hipErrorStreamCaptureUnjoined      904
#define hipErrorStreamCaptureIsolation     905
#define hipErrorStreamCaptureImplicit      906
#define hipErrorCapturedEvent              907
#define hipErrorStreamCaptureWrongThread   908
#define hipErrorGraphExecUpdateFailure     910
#define hipErrorUnknown                    999
#define hipErrorRuntimeMemory             1052
#define hipErrorRuntimeOther              1053

/* Stream flags */
#define hipStreamDefault      0x00
#define hipStreamNonBlocking  0x01

/* Event flags */
#define hipEventDefault       0x00
#define hipEventBlockingSync  0x01
#define hipEventDisableTiming 0x02
#define hipEventInterprocess  0x04

/* Memory flags */
#define hipMemAttachGlobal    0x01
#define hipMemAttachHost      0x02
#define hipHostMallocDefault  0x00
#define hipHostMallocPortable 0x01
#define hipHostMallocMapped   0x02
#define hipHostMallocWriteCombined 0x04
#define hipHostMallocCoherent 0x40000000
#define hipHostMallocNonCoherent 0x80000000

/* ============================================================================
 * HIPRTC Types
 * ============================================================================ */

typedef struct _hiprtcProgram* hiprtcProgram;
typedef enum hiprtcResult {
    HIPRTC_SUCCESS = 0,
    HIPRTC_ERROR_OUT_OF_MEMORY = 1,
    HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    HIPRTC_ERROR_INVALID_INPUT = 3,
    HIPRTC_ERROR_INVALID_PROGRAM = 4,
    HIPRTC_ERROR_INVALID_OPTION = 5,
    HIPRTC_ERROR_COMPILATION = 6,
    HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    HIPRTC_ERROR_INTERNAL_ERROR = 11
} hiprtcResult;

/* HIPRTC JIT input type */
typedef enum hiprtcJITInputType {
    HIPRTC_JIT_INPUT_CUBIN = 0,
    HIPRTC_JIT_INPUT_PTX = 1,
    HIPRTC_JIT_INPUT_FATBINARY = 2,
    HIPRTC_JIT_INPUT_OBJECT = 3,
    HIPRTC_JIT_INPUT_LIBRARY = 4,
    HIPRTC_JIT_INPUT_NVVM = 5
} hiprtcJITInputType;

/* ============================================================================
 * HIP Function Type Definitions
 * ============================================================================ */

/* Error handling */
typedef hipError_t (HIPAPI *thipGetErrorName)(hipError_t error, const char** name);
typedef hipError_t (HIPAPI *thipGetErrorString)(hipError_t error, const char** str);
typedef const char* (HIPAPI *thipGetErrorName_)(hipError_t error);
typedef const char* (HIPAPI *thipGetErrorString_)(hipError_t error);
typedef hipError_t (HIPAPI *thipGetLastError)(void);
typedef hipError_t (HIPAPI *thipPeekAtLastError)(void);

/* Initialization */
typedef hipError_t (HIPAPI *thipInit)(unsigned int flags);

/* Device management */
typedef hipError_t (HIPAPI *thipGetDeviceCount)(int* count);
typedef hipError_t (HIPAPI *thipGetDevice)(int* deviceId);
typedef hipError_t (HIPAPI *thipSetDevice)(int deviceId);
typedef hipError_t (HIPAPI *thipGetDeviceProperties)(hipDeviceProp_t* props, int deviceId);
typedef hipError_t (HIPAPI *thipDeviceGetAttribute)(int* value, hipDeviceAttribute_t attr, int deviceId);
typedef hipError_t (HIPAPI *thipDeviceGetName)(char* name, int len, int deviceId);
typedef hipError_t (HIPAPI *thipDeviceGetPCIBusId)(char* pciBusId, int len, int deviceId);
typedef hipError_t (HIPAPI *thipDeviceGetByPCIBusId)(int* deviceId, const char* pciBusId);
typedef hipError_t (HIPAPI *thipDeviceTotalMem)(size_t* bytes, int deviceId);
typedef hipError_t (HIPAPI *thipDeviceSynchronize)(void);
typedef hipError_t (HIPAPI *thipDeviceReset)(void);

/* Context management */
typedef hipError_t (HIPAPI *thipCtxCreate)(hipCtx_t* ctx, unsigned int flags, hipDevice_t device);
typedef hipError_t (HIPAPI *thipCtxDestroy)(hipCtx_t ctx);
typedef hipError_t (HIPAPI *thipCtxPushCurrent)(hipCtx_t ctx);
typedef hipError_t (HIPAPI *thipCtxPopCurrent)(hipCtx_t* ctx);
typedef hipError_t (HIPAPI *thipCtxSetCurrent)(hipCtx_t ctx);
typedef hipError_t (HIPAPI *thipCtxGetCurrent)(hipCtx_t* ctx);
typedef hipError_t (HIPAPI *thipCtxGetDevice)(hipDevice_t* device);
typedef hipError_t (HIPAPI *thipCtxSynchronize)(void);

/* Memory management */
typedef hipError_t (HIPAPI *thipMalloc)(void** ptr, size_t size);
typedef hipError_t (HIPAPI *thipMallocPitch)(void** ptr, size_t* pitch, size_t width, size_t height);
typedef hipError_t (HIPAPI *thipMalloc3D)(void** ptr, size_t* pitch, size_t width, size_t height, size_t depth);
typedef hipError_t (HIPAPI *thipFree)(void* ptr);
typedef hipError_t (HIPAPI *thipMallocHost)(void** ptr, size_t size);
typedef hipError_t (HIPAPI *thipHostMalloc)(void** ptr, size_t size, unsigned int flags);
typedef hipError_t (HIPAPI *thipFreeHost)(void* ptr);
typedef hipError_t (HIPAPI *thipHostFree)(void* ptr);
typedef hipError_t (HIPAPI *thipMallocManaged)(void** ptr, size_t size, unsigned int flags);
typedef hipError_t (HIPAPI *thipMemcpy)(void* dst, const void* src, size_t size, hipMemcpyKind kind);
typedef hipError_t (HIPAPI *thipMemcpyAsync)(void* dst, const void* src, size_t size, hipMemcpyKind kind, hipStream_t stream);
typedef hipError_t (HIPAPI *thipMemcpyHtoD)(hipDeviceptr_t dst, void* src, size_t size);
typedef hipError_t (HIPAPI *thipMemcpyDtoH)(void* dst, hipDeviceptr_t src, size_t size);
typedef hipError_t (HIPAPI *thipMemcpyDtoD)(hipDeviceptr_t dst, hipDeviceptr_t src, size_t size);
typedef hipError_t (HIPAPI *thipMemcpyHtoDAsync)(hipDeviceptr_t dst, void* src, size_t size, hipStream_t stream);
typedef hipError_t (HIPAPI *thipMemcpyDtoHAsync)(void* dst, hipDeviceptr_t src, size_t size, hipStream_t stream);
typedef hipError_t (HIPAPI *thipMemcpyDtoDAsync)(hipDeviceptr_t dst, hipDeviceptr_t src, size_t size, hipStream_t stream);
typedef hipError_t (HIPAPI *thipMemset)(void* ptr, int value, size_t count);
typedef hipError_t (HIPAPI *thipMemsetAsync)(void* ptr, int value, size_t count, hipStream_t stream);
typedef hipError_t (HIPAPI *thipMemsetD8)(hipDeviceptr_t dst, unsigned char value, size_t count);
typedef hipError_t (HIPAPI *thipMemsetD16)(hipDeviceptr_t dst, unsigned short value, size_t count);
typedef hipError_t (HIPAPI *thipMemsetD32)(hipDeviceptr_t dst, unsigned int value, size_t count);
typedef hipError_t (HIPAPI *thipMemGetInfo)(size_t* free, size_t* total);
typedef hipError_t (HIPAPI *thipMemPtrGetInfo)(void* ptr, size_t* size);

/* Module management */
typedef hipError_t (HIPAPI *thipModuleLoad)(hipModule_t* module, const char* fname);
typedef hipError_t (HIPAPI *thipModuleLoadData)(hipModule_t* module, const void* image);
typedef hipError_t (HIPAPI *thipModuleLoadDataEx)(hipModule_t* module, const void* image, unsigned int numOptions, void** options, void** optionValues);
typedef hipError_t (HIPAPI *thipModuleUnload)(hipModule_t module);
typedef hipError_t (HIPAPI *thipModuleGetFunction)(hipFunction_t* function, hipModule_t module, const char* name);
typedef hipError_t (HIPAPI *thipModuleGetGlobal)(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t module, const char* name);

/* Function management */
typedef hipError_t (HIPAPI *thipFuncGetAttribute)(int* value, hipFuncAttribute_t attrib, hipFunction_t func);
typedef hipError_t (HIPAPI *thipFuncSetAttribute)(hipFunction_t func, hipFuncAttribute_t attrib, int value);
typedef hipError_t (HIPAPI *thipFuncSetCacheConfig)(hipFunction_t func, int config);
typedef hipError_t (HIPAPI *thipFuncSetSharedMemConfig)(hipFunction_t func, int config);

/* dim3 structure for kernel launch */
#ifndef __HIP_PLATFORM_AMD__
#ifdef __cplusplus
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int _x = 1, unsigned int _y = 1, unsigned int _z = 1)
        : x(_x), y(_y), z(_z) {}
};
#else
typedef struct hip_dim3 {
    unsigned int x, y, z;
} dim3;
#endif
#endif

/* Kernel launch */
typedef hipError_t (HIPAPI *thipModuleLaunchKernel)(hipFunction_t f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, hipStream_t stream,
    void** kernelParams, void** extra);
typedef hipError_t (HIPAPI *thipLaunchKernel)(const void* func,
    dim3 gridDim, dim3 blockDim,
    void** args, size_t sharedMem, hipStream_t stream);
typedef hipError_t (HIPAPI *thipLaunchCooperativeKernel)(const void* func,
    dim3 gridDim, dim3 blockDim,
    void** args, size_t sharedMem, hipStream_t stream);

/* Stream management */
typedef hipError_t (HIPAPI *thipStreamCreate)(hipStream_t* stream);
typedef hipError_t (HIPAPI *thipStreamCreateWithFlags)(hipStream_t* stream, unsigned int flags);
typedef hipError_t (HIPAPI *thipStreamCreateWithPriority)(hipStream_t* stream, unsigned int flags, int priority);
typedef hipError_t (HIPAPI *thipStreamDestroy)(hipStream_t stream);
typedef hipError_t (HIPAPI *thipStreamQuery)(hipStream_t stream);
typedef hipError_t (HIPAPI *thipStreamSynchronize)(hipStream_t stream);
typedef hipError_t (HIPAPI *thipStreamWaitEvent)(hipStream_t stream, hipEvent_t event, unsigned int flags);
typedef hipError_t (HIPAPI *thipStreamGetFlags)(hipStream_t stream, unsigned int* flags);
typedef hipError_t (HIPAPI *thipStreamGetPriority)(hipStream_t stream, int* priority);

/* Event management */
typedef hipError_t (HIPAPI *thipEventCreate)(hipEvent_t* event);
typedef hipError_t (HIPAPI *thipEventCreateWithFlags)(hipEvent_t* event, unsigned int flags);
typedef hipError_t (HIPAPI *thipEventDestroy)(hipEvent_t event);
typedef hipError_t (HIPAPI *thipEventRecord)(hipEvent_t event, hipStream_t stream);
typedef hipError_t (HIPAPI *thipEventQuery)(hipEvent_t event);
typedef hipError_t (HIPAPI *thipEventSynchronize)(hipEvent_t event);
typedef hipError_t (HIPAPI *thipEventElapsedTime)(float* ms, hipEvent_t start, hipEvent_t stop);

/* Occupancy */
typedef hipError_t (HIPAPI *thipOccupancyMaxActiveBlocksPerMultiprocessor)(int* numBlocks, const void* func, int blockSize, size_t dynSharedMemPerBlk);
typedef hipError_t (HIPAPI *thipOccupancyMaxPotentialBlockSize)(int* gridSize, int* blockSize, const void* func, size_t dynSharedMemPerBlk, int blockSizeLimit);

/* ============================================================================
 * HIPRTC Function Type Definitions
 * ============================================================================ */

typedef const char* (HIPAPI *thiprtcGetErrorString)(hiprtcResult result);
typedef hiprtcResult (HIPAPI *thiprtcVersion)(int* major, int* minor);
typedef hiprtcResult (HIPAPI *thiprtcCreateProgram)(hiprtcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames);
typedef hiprtcResult (HIPAPI *thiprtcDestroyProgram)(hiprtcProgram* prog);
typedef hiprtcResult (HIPAPI *thiprtcCompileProgram)(hiprtcProgram prog, int numOptions, const char** options);
typedef hiprtcResult (HIPAPI *thiprtcGetProgramLogSize)(hiprtcProgram prog, size_t* logSizeRet);
typedef hiprtcResult (HIPAPI *thiprtcGetProgramLog)(hiprtcProgram prog, char* log);
typedef hiprtcResult (HIPAPI *thiprtcGetCodeSize)(hiprtcProgram prog, size_t* codeSizeRet);
typedef hiprtcResult (HIPAPI *thiprtcGetCode)(hiprtcProgram prog, char* code);
typedef hiprtcResult (HIPAPI *thiprtcGetBitcodeSize)(hiprtcProgram prog, size_t* bitcodeSizeRet);
typedef hiprtcResult (HIPAPI *thiprtcGetBitcode)(hiprtcProgram prog, char* bitcode);
typedef hiprtcResult (HIPAPI *thiprtcAddNameExpression)(hiprtcProgram prog, const char* nameExpression);
typedef hiprtcResult (HIPAPI *thiprtcGetLoweredName)(hiprtcProgram prog, const char* nameExpression, const char** loweredName);

/* ============================================================================
 * HIP Function Declarations (extern globals)
 * ============================================================================ */

/* Error handling */
extern thipGetErrorName hipGetErrorName;
extern thipGetErrorString hipGetErrorString;
extern thipGetLastError hipGetLastError;
extern thipPeekAtLastError hipPeekAtLastError;

/* Initialization */
extern thipInit hipInit;

/* Device management */
extern thipGetDeviceCount hipGetDeviceCount;
extern thipGetDevice hipGetDevice;
extern thipSetDevice hipSetDevice;
extern thipGetDeviceProperties hipGetDeviceProperties;
extern thipDeviceGetAttribute hipDeviceGetAttribute;
extern thipDeviceGetName hipDeviceGetName;
extern thipDeviceGetPCIBusId hipDeviceGetPCIBusId;
extern thipDeviceGetByPCIBusId hipDeviceGetByPCIBusId;
extern thipDeviceTotalMem hipDeviceTotalMem;
extern thipDeviceSynchronize hipDeviceSynchronize;
extern thipDeviceReset hipDeviceReset;

/* Context management */
extern thipCtxCreate hipCtxCreate;
extern thipCtxDestroy hipCtxDestroy;
extern thipCtxPushCurrent hipCtxPushCurrent;
extern thipCtxPopCurrent hipCtxPopCurrent;
extern thipCtxSetCurrent hipCtxSetCurrent;
extern thipCtxGetCurrent hipCtxGetCurrent;
extern thipCtxGetDevice hipCtxGetDevice;
extern thipCtxSynchronize hipCtxSynchronize;

/* Memory management */
extern thipMalloc hipMalloc;
extern thipMallocPitch hipMallocPitch;
extern thipMalloc3D hipMalloc3D;
extern thipFree hipFree;
extern thipMallocHost hipMallocHost;
extern thipHostMalloc hipHostMalloc;
extern thipFreeHost hipFreeHost;
extern thipHostFree hipHostFree;
extern thipMallocManaged hipMallocManaged;
extern thipMemcpy hipMemcpy;
extern thipMemcpyAsync hipMemcpyAsync;
extern thipMemcpyHtoD hipMemcpyHtoD;
extern thipMemcpyDtoH hipMemcpyDtoH;
extern thipMemcpyDtoD hipMemcpyDtoD;
extern thipMemcpyHtoDAsync hipMemcpyHtoDAsync;
extern thipMemcpyDtoHAsync hipMemcpyDtoHAsync;
extern thipMemcpyDtoDAsync hipMemcpyDtoDAsync;
extern thipMemset hipMemset;
extern thipMemsetAsync hipMemsetAsync;
extern thipMemsetD8 hipMemsetD8;
extern thipMemsetD16 hipMemsetD16;
extern thipMemsetD32 hipMemsetD32;
extern thipMemGetInfo hipMemGetInfo;
extern thipMemPtrGetInfo hipMemPtrGetInfo;

/* Module management */
extern thipModuleLoad hipModuleLoad;
extern thipModuleLoadData hipModuleLoadData;
extern thipModuleLoadDataEx hipModuleLoadDataEx;
extern thipModuleUnload hipModuleUnload;
extern thipModuleGetFunction hipModuleGetFunction;
extern thipModuleGetGlobal hipModuleGetGlobal;

/* Function management */
extern thipFuncGetAttribute hipFuncGetAttribute;
extern thipFuncSetAttribute hipFuncSetAttribute;
extern thipFuncSetCacheConfig hipFuncSetCacheConfig;
extern thipFuncSetSharedMemConfig hipFuncSetSharedMemConfig;

/* Kernel launch */
extern thipModuleLaunchKernel hipModuleLaunchKernel;

/* Stream management */
extern thipStreamCreate hipStreamCreate;
extern thipStreamCreateWithFlags hipStreamCreateWithFlags;
extern thipStreamCreateWithPriority hipStreamCreateWithPriority;
extern thipStreamDestroy hipStreamDestroy;
extern thipStreamQuery hipStreamQuery;
extern thipStreamSynchronize hipStreamSynchronize;
extern thipStreamWaitEvent hipStreamWaitEvent;
extern thipStreamGetFlags hipStreamGetFlags;
extern thipStreamGetPriority hipStreamGetPriority;

/* Event management */
extern thipEventCreate hipEventCreate;
extern thipEventCreateWithFlags hipEventCreateWithFlags;
extern thipEventDestroy hipEventDestroy;
extern thipEventRecord hipEventRecord;
extern thipEventQuery hipEventQuery;
extern thipEventSynchronize hipEventSynchronize;
extern thipEventElapsedTime hipEventElapsedTime;

/* Occupancy */
extern thipOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor;
extern thipOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize;

/* ============================================================================
 * HIPRTC Function Declarations (extern globals)
 * ============================================================================ */

extern thiprtcGetErrorString hiprtcGetErrorString;
extern thiprtcVersion hiprtcVersion;
extern thiprtcCreateProgram hiprtcCreateProgram;
extern thiprtcDestroyProgram hiprtcDestroyProgram;
extern thiprtcCompileProgram hiprtcCompileProgram;
extern thiprtcGetProgramLogSize hiprtcGetProgramLogSize;
extern thiprtcGetProgramLog hiprtcGetProgramLog;
extern thiprtcGetCodeSize hiprtcGetCodeSize;
extern thiprtcGetCode hiprtcGetCode;
extern thiprtcGetBitcodeSize hiprtcGetBitcodeSize;
extern thiprtcGetBitcode hiprtcGetBitcode;
extern thiprtcAddNameExpression hiprtcAddNameExpression;
extern thiprtcGetLoweredName hiprtcGetLoweredName;

/* ============================================================================
 * ROCEW Initialization
 * ============================================================================ */

/**
 * Initialize ROCEW library.
 * @param flags Combination of ROCEW_INIT_* flags
 * @return ROCEW_SUCCESS on success, error code otherwise
 */
int rocewInit(unsigned int flags);

/**
 * Check if HIP runtime is available.
 * @return 1 if available, 0 otherwise
 */
int rocewHipAvailable(void);

/**
 * Check if HIPRTC is available.
 * @return 1 if available, 0 otherwise
 */
int rocewHiprtcAvailable(void);

/**
 * Get architecture string for RDNA4.
 * Returns "gfx1200" or "gfx1201" depending on specific RDNA4 variant.
 */
const char* rocewGetRDNA4ArchString(int deviceId);

/**
 * Check if device supports RDNA4 architecture.
 * @return 1 if RDNA4, 0 otherwise
 */
int rocewIsRDNA4(int deviceId);

#ifdef __cplusplus
}
#endif

#endif /* ROCEW_H_ */
