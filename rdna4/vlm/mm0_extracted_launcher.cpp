/*
 * mm0_extracted_launcher.cpp
 *
 * Standalone launcher that invokes hipBLASLt's algo-73624 kernel directly,
 * without linking against libhipblaslt at runtime. The kernel binary is
 * loaded from hipBLASLt's TensileLibrary code object via hipModuleLoad,
 * and the 140-byte kernarg buffer is constructed by this file using values
 * that were captured (via dump_kernarg_shim.c) from a real hipBLASLt run
 * for the mm0 problem (M=1024, N=4608, K=4608, BF16 in / FP32 out, NT).
 *
 * Captured kernarg metadata:
 *   grid (work items) = (4608, 8, 1)   = 36 × 8 = 288 workgroups
 *   block             = (128, 1, 1)    (32 lanes × 4 waves, flattened)
 *   shared_mem_dyn    = 0              (LDS = 26624B is .group_segment_fixed_size)
 *   args_size         = 140            (.kernarg_segment_size = 144 with pad)
 *
 * Captured kernarg header values (constant for this problem shape):
 *   gemm_info     = 1            (gemmCount=1, argType=inline)
 *   internalArg0  = 0x02200001   (gsu=1, staggerU bits)
 *   internalArg1  = 0x08010008   (wgmxccg=32, wgmxcc=1, wgm=8)
 *   numWG         = 36           (X-dim workgroup count)
 *
 * Drop-in replacement for the mm0_hipblaslt_bridge.cpp interface so that
 * bench_vlm_gemm.c's `mm0blaslt` mode (or a new `mm0extract` mode) can
 * exercise it identically.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

// rocew's dynamic-loader wrapper defines hipModuleLoad/GetFunction/Unload as
// global function-pointer variables in BSS that collide with libamdhip64 at
// link time — the bench links rocew first so calls in this TU resolve to
// rocew's pointers (NULL until rocewInit fills them, and even then NOT
// guaranteed). Bypass entirely by resolving through dlsym(RTLD_DEFAULT, ...)
// which finds the real libamdhip64 symbols since libamdhip64 is in the link.
// hipExtModuleLaunchKernel is undefined in rocew so it resolves normally.
typedef hipError_t (*pf_hipModuleLoad)(hipModule_t*, const char*);
typedef hipError_t (*pf_hipModuleGetFunction)(hipFunction_t*, hipModule_t,
                                              const char*);
typedef hipError_t (*pf_hipModuleUnload)(hipModule_t);
typedef hipError_t (*pf_hipStreamCreate)(hipStream_t*);
typedef hipError_t (*pf_hipStreamDestroy)(hipStream_t);

pf_hipModuleLoad        s_hipModuleLoad        = nullptr;
pf_hipModuleGetFunction s_hipModuleGetFunction = nullptr;
pf_hipModuleUnload      s_hipModuleUnload      = nullptr;
pf_hipStreamCreate      s_hipStreamCreate      = nullptr;
pf_hipStreamDestroy     s_hipStreamDestroy     = nullptr;

bool resolve_hip_syms() {
  if (s_hipModuleLoad && s_hipModuleGetFunction && s_hipModuleUnload &&
      s_hipStreamCreate && s_hipStreamDestroy) return true;
  s_hipModuleLoad = (pf_hipModuleLoad)dlsym(RTLD_DEFAULT, "hipModuleLoad");
  s_hipModuleGetFunction =
      (pf_hipModuleGetFunction)dlsym(RTLD_DEFAULT, "hipModuleGetFunction");
  s_hipModuleUnload =
      (pf_hipModuleUnload)dlsym(RTLD_DEFAULT, "hipModuleUnload");
  s_hipStreamCreate  = (pf_hipStreamCreate)dlsym(RTLD_DEFAULT, "hipStreamCreate");
  s_hipStreamDestroy = (pf_hipStreamDestroy)dlsym(RTLD_DEFAULT, "hipStreamDestroy");
  if (!s_hipModuleLoad || !s_hipModuleGetFunction || !s_hipModuleUnload ||
      !s_hipStreamCreate || !s_hipStreamDestroy) {
    std::fprintf(stderr,
                 "[mm0_extracted] dlsym failed: load=%p getfn=%p unload=%p "
                 "scre=%p sdes=%p\n",
                 (void*)s_hipModuleLoad, (void*)s_hipModuleGetFunction,
                 (void*)s_hipModuleUnload, (void*)s_hipStreamCreate,
                 (void*)s_hipStreamDestroy);
    return false;
  }
  return true;
}

// Selected after exhaustive sweep of 59 DTVB1+PGR2 variants (2026-04-30):
// LDSB1+CLR0+LBSPPA256+TLDS2+SS1+SVW4+VWA4+VWB4 sustains ~172 TFLOP/s
// (88% peak) on this shape vs 155 for the original algo 73624 default.
const char* KERNEL_SYM =
    "Cijk_Alik_Bljk_BSS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_"
    "LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB1_EPS0_"
    "FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA1201_IU1_K1_LDSTI0_LBSPPA256_"
    "LBSPPB0_LBSPPM0_LPA16_LPB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT4_4_MO40_NTn1_"
    "NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB2_ONLL1_PGR2_PLR1_PKA0_SIA3_"
    "SS1_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_TLDS2_ULSGRO0_USL1_UIOFGRO0_"
    "USFGROn1_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS32_WG32_4_1";

const char* DEFAULT_CO_PATH =
    "/opt/rocm-7.2.1/lib/hipblaslt/library/"
    "TensileLibrary_BB_SB_HA_Bias_SAV_UA_Type_BS_HPA_Contraction_l_Alik_Bljk_"
    "Cijk_Dijk_gfx1201.co";

// Kernarg field offsets per .args metadata of the kernel.
constexpr size_t OFF_GEMM_INFO    = 0;
constexpr size_t OFF_INTERNAL0    = 4;
constexpr size_t OFF_INTERNAL1    = 8;
constexpr size_t OFF_NUMWG        = 12;
constexpr size_t OFF_SIZES_FREE0  = 16;   // N
constexpr size_t OFF_SIZES_FREE1  = 20;   // M
constexpr size_t OFF_SIZES_FREE2  = 24;   // batch
constexpr size_t OFF_SIZES_SUM0   = 28;   // K
constexpr size_t OFF_D            = 32;
constexpr size_t OFF_C            = 40;
constexpr size_t OFF_A            = 48;
constexpr size_t OFF_B            = 56;
constexpr size_t OFF_STRIDE_D0    = 64;   // leading dim of D
constexpr size_t OFF_STRIDE_D1    = 68;   // batch stride
constexpr size_t OFF_STRIDE_C0    = 72;
constexpr size_t OFF_STRIDE_C1    = 76;
constexpr size_t OFF_STRIDE_A0    = 80;
constexpr size_t OFF_STRIDE_A1    = 84;
constexpr size_t OFF_STRIDE_B0    = 88;
constexpr size_t OFF_STRIDE_B1    = 92;
constexpr size_t OFF_ALPHA        = 96;
constexpr size_t OFF_BETA         = 100;
constexpr size_t OFF_SCALE_AV     = 104;
constexpr size_t OFF_BIAS         = 112;
constexpr size_t OFF_BIAS_TYPE    = 120;
constexpr size_t OFF_STRIDE_BIAS  = 124;
constexpr size_t OFF_ACT_ALPHA    = 128;
constexpr size_t OFF_ACT_BETA     = 132;
constexpr size_t OFF_ACT_TYPE     = 136;
constexpr size_t KERNARG_BYTES    = 140;
constexpr size_t KERNARG_SEG_SIZE = 144;

#define HIP_RET(expr)                                                          \
  do {                                                                         \
    hipError_t _err = (expr);                                                  \
    if (_err != hipSuccess) {                                                  \
      std::fprintf(stderr, "[mm0_extracted] HIP %s:%d: %s\n", __FILE__,       \
                   __LINE__, hipGetErrorString(_err));                         \
      return -1;                                                               \
    }                                                                          \
  } while (0)

template <typename T>
inline void put(uint8_t* p, size_t off, T v) {
  std::memcpy(p + off, &v, sizeof(T));
}

struct State {
  hipModule_t   module = nullptr;
  hipFunction_t func   = nullptr;
  hipStream_t   stream = nullptr;
  int           M = 0, N = 0, K = 0;
  uint8_t       kernarg[KERNARG_SEG_SIZE]{};
};

State g_state;

}  // namespace

extern "C" int mm0_extracted_init(int M, int N, int K, const void* d_bias) {
  if (g_state.module != nullptr) {
    std::fprintf(stderr, "[mm0_extracted] already initialized\n");
    return -1;
  }
  if (M != 1024 || N != 4608 || K != 4608) {
    std::fprintf(stderr,
                 "[mm0_extracted] only M=1024 N=4608 K=4608 supported "
                 "(got M=%d N=%d K=%d). Re-capture kernarg for new shape.\n",
                 M, N, K);
    return -1;
  }
  g_state.M = M;
  g_state.N = N;
  g_state.K = K;

  const char* co_path = std::getenv("MM0_EXTRACTED_CO");
  if (!co_path || !co_path[0]) {
    co_path = DEFAULT_CO_PATH;
  }

  if (!resolve_hip_syms()) return -1;

  HIP_RET(s_hipModuleLoad(&g_state.module, co_path));

  // Override kernel symbol via MM0_EXTRACTED_KERNEL_SYM (full mangled name).
  // Useful for testing alternative algos (e.g. DTVA+DTVB, PGR2 variants).
  const char* sym = std::getenv("MM0_EXTRACTED_KERNEL_SYM");
  if (!sym || !sym[0]) sym = KERNEL_SYM;

  hipError_t err = s_hipModuleGetFunction(&g_state.func, g_state.module, sym);
  if (err != hipSuccess) {
    std::fprintf(stderr,
                 "[mm0_extracted] hipModuleGetFunction(%s) failed: %s\n",
                 sym, hipGetErrorString(err));
    s_hipModuleUnload(g_state.module);
    g_state.module = nullptr;
    return -1;
  }

  std::memset(g_state.kernarg, 0, sizeof g_state.kernarg);
  put<uint32_t>(g_state.kernarg, OFF_GEMM_INFO,   1u);
  put<uint32_t>(g_state.kernarg, OFF_INTERNAL0,   0x02200001u);
  put<uint32_t>(g_state.kernarg, OFF_INTERNAL1,   0x08010008u);
  put<uint32_t>(g_state.kernarg, OFF_NUMWG,       36u);
  put<uint32_t>(g_state.kernarg, OFF_SIZES_FREE0, static_cast<uint32_t>(N));
  put<uint32_t>(g_state.kernarg, OFF_SIZES_FREE1, static_cast<uint32_t>(M));
  put<uint32_t>(g_state.kernarg, OFF_SIZES_FREE2, 1u);
  put<uint32_t>(g_state.kernarg, OFF_SIZES_SUM0,  static_cast<uint32_t>(K));
  put<uint32_t>(g_state.kernarg, OFF_STRIDE_D0,   static_cast<uint32_t>(N));
  put<uint32_t>(g_state.kernarg, OFF_STRIDE_D1,   0u);
  put<uint32_t>(g_state.kernarg, OFF_STRIDE_C0,   static_cast<uint32_t>(N));
  put<uint32_t>(g_state.kernarg, OFF_STRIDE_C1,   0u);
  put<uint32_t>(g_state.kernarg, OFF_STRIDE_A0,   static_cast<uint32_t>(K));
  put<uint32_t>(g_state.kernarg, OFF_STRIDE_A1,   0u);
  put<uint32_t>(g_state.kernarg, OFF_STRIDE_B0,   static_cast<uint32_t>(K));
  put<uint32_t>(g_state.kernarg, OFF_STRIDE_B1,   0u);
  put<float>   (g_state.kernarg, OFF_ALPHA, 1.0f);
  put<float>   (g_state.kernarg, OFF_BETA,  0.0f);
  put<void*>   (g_state.kernarg, OFF_SCALE_AV, nullptr);
  put<const void*>(g_state.kernarg, OFF_BIAS, d_bias);
  put<uint32_t>(g_state.kernarg, OFF_BIAS_TYPE,   0u);
  put<uint32_t>(g_state.kernarg, OFF_STRIDE_BIAS, 0u);
  put<float>   (g_state.kernarg, OFF_ACT_ALPHA, 0.0f);
  put<float>   (g_state.kernarg, OFF_ACT_BETA,  0.0f);
  put<uint32_t>(g_state.kernarg, OFF_ACT_TYPE, 0u);

  // Optionally use a dedicated non-default stream. Default stream (NULL)
  // is the legacy system-wide-serialized stream which can incur extra
  // synchronization vs a non-default stream.
  if (std::getenv("MM0_EXTRACTED_NONDEFSTREAM")) {
    HIP_RET(s_hipStreamCreate(&g_state.stream));
  }

  std::fprintf(stderr,
               "[mm0_extracted] init M=%d N=%d K=%d sym ok co=%s bias=%s stream=%s\n",
               M, N, K, co_path, d_bias ? "yes" : "no",
               g_state.stream ? "non-default" : "default");
  return 0;
}

extern "C" int mm0_extracted_run(void* d_y, const void* d_w, const void* d_x) {
  if (g_state.func == nullptr) return -1;

  put<void*>(g_state.kernarg, OFF_D, d_y);
  put<void*>(g_state.kernarg, OFF_C, d_y);
  put<const void*>(g_state.kernarg, OFF_A, d_w);
  put<const void*>(g_state.kernarg, OFF_B, d_x);

  // Use the segment-padded size (144) by default. The 140-byte captured size
  // matches the kernel's .args metadata exactly, but some HIP runtime paths
  // align kernarg uploads to the .kernarg_segment_size which is 144.
  // Override via MM0_EXTRACTED_ARGSZ=140|144.
  static size_t s_arg_size = []() {
    const char* e = std::getenv("MM0_EXTRACTED_ARGSZ");
    return (e && std::atoi(e) >= 140) ? (size_t)std::atoi(e) : KERNARG_BYTES;
  }();
  size_t arg_size = s_arg_size;
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, g_state.kernarg,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE,    &arg_size,
                    HIP_LAUNCH_PARAM_END};

  HIP_RET(hipExtModuleLaunchKernel(
      g_state.func,
      /*globalX=*/4608, /*globalY=*/8, /*globalZ=*/1,
      /*localX =*/128,  /*localY=*/1, /*localZ=*/1,
      /*sharedMem=*/0, /*stream=*/g_state.stream,
      /*kernelParams=*/nullptr, (void**)&config,
      /*startEvent=*/nullptr, /*stopEvent=*/nullptr));
  return 0;
}

extern "C" hipStream_t mm0_extracted_get_stream(void) {
  return g_state.stream;
}

extern "C" int mm0_extracted_destroy(void) {
  if (g_state.stream) {
    if (s_hipStreamDestroy) s_hipStreamDestroy(g_state.stream);
    g_state.stream = nullptr;
  }
  if (g_state.module) {
    if (s_hipModuleUnload) s_hipModuleUnload(g_state.module);
    g_state.module = nullptr;
  }
  g_state.func = nullptr;
  return 0;
}
