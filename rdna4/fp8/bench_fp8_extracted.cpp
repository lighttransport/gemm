/*
 * bench_fp8_extracted.cpp
 *
 * Standalone FP8 GEMM bench that calls the extracted hipBLASLt FP8 kernel
 * directly via hipModuleLoad + hipExtModuleLaunchKernel — no link against
 * libhipblaslt at runtime.
 *
 * Kernel:
 *   Cijk_Alik_Bljk_F8SS_..._MT128x128x64_MI16x16x1_..._MIWT8_2_..._WG16_8_1
 *   from TensileLibrary_F8F8_SF8_HA_Bias_SAB_SCD_SAV_UA_Type_F8S_HPA
 *        _Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1201.co
 *
 * Layout (op-A=T, op-B=N): D[M,N] = A[K,M]^T * B[K,N], all row-major.
 * For mm0 (M=1024, N=4608, K=4608):
 *   - kernarg buffer is 172 bytes (segment 176)
 *   - grid (work-items) = (4608, 8, 1), block = (128, 1, 1)
 *   - LDS dynamic = 0 (group_segment_fixed_size handled in .kd)
 *   - internalArg0 = 0x02200001, internalArg1 = 0x08010008, numWG = 36
 *
 * Captured via dump_kernarg_shim from a real hipBLASLt run on this shape.
 * Identified the fastest dispatched algo via rocprofv3 kernel-trace.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_ext.h>

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

namespace {

const char* DEFAULT_CO_PATH = "rdna4/fp8/fp8_kernel_gfx1201.co";

// Kernel for algo[9] in bench_hipblaslt_fp8 — MIWT4_4 PGR2, the algo whose
// kernarg we captured via dump_kernarg_shim.
const char* DEFAULT_KERNEL_SYM =
    "Cijk_Alik_Bljk_F8SS_BH_Bias_SHB_HA_S_SAB_SCD_SAV_UserArgs_"
    "MT128x128x64_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_"
    "DTLA0_DTLB0_DTVA0_DTVB1_EPS0_FDSI0_GRPM1_GRVWA16_GRVWB16_GSUAMB_GLS0_"
    "ISA1201_IU1_K1_LDSTI0_LBSPPA256_LBSPPB0_LBSPPM0_LPA32_LPB0_LPM0_LRVW16_"
    "LWPMn1_MIAV1_MIWT4_4_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_"
    "NLCB2_ONLL0_PGR2_PLR1_PKA0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_"
    "SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA4_VWB4_WSGRA0_"
    "WSGRB0_WS32_WG32_4_1";

// FP8 kernarg layout (172B used, 176B padded segment).
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
constexpr size_t OFF_STRIDE_D0    = 64;
constexpr size_t OFF_STRIDE_D1    = 68;
constexpr size_t OFF_STRIDE_C0    = 72;
constexpr size_t OFF_STRIDE_C1    = 76;
constexpr size_t OFF_STRIDE_A0    = 80;
constexpr size_t OFF_STRIDE_A1    = 84;
constexpr size_t OFF_STRIDE_B0    = 88;
constexpr size_t OFF_STRIDE_B1    = 92;
constexpr size_t OFF_ALPHA        = 96;
constexpr size_t OFF_BETA         = 100;
constexpr size_t OFF_SCALE_A      = 104;
constexpr size_t OFF_SCALE_B      = 112;
constexpr size_t OFF_SCALE_C      = 120;
constexpr size_t OFF_SCALE_D      = 128;
constexpr size_t OFF_SCALE_AV     = 136;
constexpr size_t OFF_BIAS         = 144;
constexpr size_t OFF_BIAS_TYPE    = 152;
constexpr size_t OFF_STRIDE_BIAS  = 156;
constexpr size_t OFF_ACT_ALPHA    = 160;
constexpr size_t OFF_ACT_BETA     = 164;
constexpr size_t OFF_ACT_TYPE     = 168;
constexpr size_t KERNARG_BYTES    = 172;
constexpr size_t KERNARG_SEG_SIZE = 176;

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t _err = (expr);                                                  \
    if (_err != hipSuccess) {                                                  \
      std::fprintf(stderr, "HIP %s:%d: %s\n", __FILE__, __LINE__,              \
                   hipGetErrorString(_err));                                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

template <typename T>
inline void put(uint8_t* p, size_t off, T v) {
  std::memcpy(p + off, &v, sizeof(T));
}

// Convert float to FP8 e4m3 (e4m3fn, no inf, NaN at 0xff/0x7f).
uint8_t fp32_to_fp8_e4m3(float f) {
  if (std::isnan(f)) return 0x7f;
  uint32_t u; std::memcpy(&u, &f, 4);
  uint32_t sign = (u >> 31) & 1;
  int32_t exp = (int32_t)((u >> 23) & 0xff) - 127;
  uint32_t mant = u & 0x7fffff;
  // Clamp to e4m3 range: max = 448 = (S=0, E=15, M=6) -> 1.75 * 2^8
  // Saturate to max-finite (NaN encoding 0x7f reserved)
  if (exp >= 8) {
    if (exp > 8 || mant > (6u << 20)) {
      return (uint8_t)((sign << 7) | 0x7e);  // ±max-finite
    }
  }
  if (exp < -9) return (uint8_t)(sign << 7);  // underflow
  // Mantissa: top 3 bits, with rounding (round-to-nearest-even).
  // Subnormal: exp < -6 -> shift mantissa right by (-6 - exp + 1)
  int e_out;
  uint32_t m_out;
  if (exp < -6) {
    int shift = -6 - exp + 1;  // bits to shift mantissa (in 24-bit form: 1.mmm)
    uint32_t m = (mant | 0x800000) >> shift;  // implicit 1
    e_out = 0;
    m_out = m;
  } else {
    e_out = exp + 7;  // bias 7
    m_out = mant | 0x800000;  // implicit 1 in bit 23
    m_out >>= 21;             // keep top 3 mantissa bits + 1 implicit
    m_out &= 0x7;             // strip implicit
  }
  // (Simple rounding: round half-to-nearest-even by inspecting bit below)
  uint32_t round_mask = 0;
  if (exp >= -6) {
    round_mask = (mant >> 20) & 1;  // bit just below kept
  }
  uint32_t out = (sign << 7) | ((e_out & 0xf) << 3) | (m_out & 0x7);
  if (round_mask) out += 1;  // simple round-up; not strict ties-to-even
  if ((out & 0x7f) >= 0x7f) out = (sign << 7) | 0x7e;  // saturate
  return (uint8_t)out;
}

float fp8_e4m3_to_fp32(uint8_t b) {
  uint32_t sign = (b >> 7) & 1;
  uint32_t e = (b >> 3) & 0xf;
  uint32_t m = b & 0x7;
  if (e == 0xf && m == 0x7) {
    return std::nanf("");  // 0x7f / 0xff = NaN
  }
  float f;
  if (e == 0) {
    if (m == 0) {
      f = 0.0f;
    } else {
      f = std::ldexp((float)m / 8.0f, -6);
    }
  } else {
    f = std::ldexp(1.0f + (float)m / 8.0f, (int)e - 7);
  }
  return sign ? -f : f;
}

}  // namespace

int main(int argc, char** argv) {
  int M = 1024, N = 4608, K = 4608;
  int iters = 100;
  bool check = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--shape") && i + 1 < argc) {
      const char* s = argv[++i];
      if (!std::strcmp(s, "mm0")) { M=1024; N=4608; K=4608; }
      else { std::fprintf(stderr, "only --shape mm0 supported\n"); return 1; }
    } else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--check")) {
      check = true;
    }
  }

  const char* co_path = std::getenv("FP8_EXTRACTED_CO");
  if (!co_path || !*co_path) co_path = DEFAULT_CO_PATH;
  const char* sym = std::getenv("FP8_EXTRACTED_KERNEL_SYM");
  if (!sym || !*sym) sym = DEFAULT_KERNEL_SYM;

  hipModule_t module;
  hipFunction_t func;
  // Try the configured path first; fall back to ./fp8_kernel_gfx1201.co so the
  // bench works whether invoked from repo root or rdna4/fp8.
  const char* fallback = "fp8_kernel_gfx1201.co";
  if (FILE* fp = std::fopen(co_path, "rb")) {
    std::fclose(fp);
  } else if (FILE* fp2 = std::fopen(fallback, "rb")) {
    std::fclose(fp2);
    co_path = fallback;
  } else {
    std::fprintf(stderr,
                 "error: kernel code object not found at %s or ./%s\n"
                 "       run rdna4/fp8/extract_fp8_kernel.sh first\n"
                 "       (or `make -C rdna4/fp8 fp8_kernel_gfx1201.co`).\n",
                 DEFAULT_CO_PATH, fallback);
    return 1;
  }
  HIP_CHECK(hipModuleLoad(&module, co_path));
  HIP_CHECK(hipModuleGetFunction(&func, module, sym));
  std::fprintf(stderr, "loaded %s\n  sym head: %.80s...\n", co_path, sym);

  // Layout (Tensile Alik_Bljk = opA=T, opB=N):
  //   A is stored [M, K] row-major (lda = strideA0 = K)
  //   B is stored [N, K] row-major (lda = strideB0 = K)
  //   D is [M, N] row-major.
  // Random FP8 bytes — decode for reference if --check.
  std::vector<uint8_t> hA((size_t)M * K), hB((size_t)N * K);
  std::srand(0xCAFE);
  // Restrict to small magnitude (e<=2) to keep accumulator from overflowing.
  for (auto& b : hA) b = (uint8_t)(((std::rand() & 0xff) & 0x37) | ((std::rand()&1) ? 0x80 : 0));
  for (auto& b : hB) b = (uint8_t)(((std::rand() & 0xff) & 0x37) | ((std::rand()&1) ? 0x80 : 0));
  std::fprintf(stderr, "host data ready\n");

  std::vector<float> hRef, hOut(M * N);
  if (check) {
    std::vector<float> Aq((size_t)M * K), Bq((size_t)N * K);
    for (size_t i = 0; i < Aq.size(); ++i) Aq[i] = fp8_e4m3_to_fp32(hA[i]);
    for (size_t i = 0; i < Bq.size(); ++i) Bq[i] = fp8_e4m3_to_fp32(hB[i]);
    hRef.assign((size_t)M * N, 0);
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        float acc = 0;
        for (int k = 0; k < K; ++k) acc += Aq[m * K + k] * Bq[n * K + k];
        hRef[m * N + n] = acc;
      }
    std::fprintf(stderr, "ref computed\n");
  }

  uint8_t *dA, *dB; float *dD;
  HIP_CHECK(hipMalloc(&dA, (size_t)M * K));
  HIP_CHECK(hipMalloc(&dB, (size_t)N * K));
  HIP_CHECK(hipMalloc(&dD, (size_t)M * N * sizeof(float)));
  HIP_CHECK(hipMemcpy(dA, hA.data(), (size_t)M*K, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, hB.data(), (size_t)N*K, hipMemcpyHostToDevice));

  // Scale tensors (1.0 each) on device — kernel reads them.
  float h_one = 1.0f;
  float *dScaleA, *dScaleB, *dScaleC, *dScaleD;
  HIP_CHECK(hipMalloc(&dScaleA, sizeof(float)));
  HIP_CHECK(hipMalloc(&dScaleB, sizeof(float)));
  HIP_CHECK(hipMalloc(&dScaleC, sizeof(float)));
  HIP_CHECK(hipMalloc(&dScaleD, sizeof(float)));
  HIP_CHECK(hipMemcpy(dScaleA, &h_one, 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dScaleB, &h_one, 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dScaleC, &h_one, 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dScaleD, &h_one, 4, hipMemcpyHostToDevice));

  // Build kernarg. If FP8_EXTRACTED_KERNARG points to a captured .bin file,
  // load it as-is (172 bytes from dump_kernarg_shim) and only patch pointers.
  uint8_t kernarg[KERNARG_SEG_SIZE]{};
  const char* kpath = std::getenv("FP8_EXTRACTED_KERNARG");
  if (kpath && *kpath) {
    FILE* fp = std::fopen(kpath, "rb");
    if (!fp) { std::fprintf(stderr, "open %s failed\n", kpath); return 1; }
    size_t r = std::fread(kernarg, 1, KERNARG_BYTES, fp);
    std::fclose(fp);
    std::fprintf(stderr, "loaded captured kernarg %zuB from %s\n", r, kpath);
  } else {
  put<uint32_t>(kernarg, OFF_GEMM_INFO,   1u);
  put<uint32_t>(kernarg, OFF_INTERNAL0,   0x02200001u);
  put<uint32_t>(kernarg, OFF_INTERNAL1,   0x08010008u);
  put<uint32_t>(kernarg, OFF_NUMWG,       (uint32_t)(N / 128));  // 36 for N=4608
  put<uint32_t>(kernarg, OFF_SIZES_FREE0, (uint32_t)N);
  put<uint32_t>(kernarg, OFF_SIZES_FREE1, (uint32_t)M);
  put<uint32_t>(kernarg, OFF_SIZES_FREE2, 1u);
  put<uint32_t>(kernarg, OFF_SIZES_SUM0,  (uint32_t)K);
  put<void*>   (kernarg, OFF_D, dD);
  put<void*>   (kernarg, OFF_C, dD);
  put<void*>   (kernarg, OFF_A, dA);
  put<void*>   (kernarg, OFF_B, dB);
  put<uint32_t>(kernarg, OFF_STRIDE_D0, (uint32_t)N);
  put<uint32_t>(kernarg, OFF_STRIDE_D1, 0u);
  put<uint32_t>(kernarg, OFF_STRIDE_C0, (uint32_t)N);
  put<uint32_t>(kernarg, OFF_STRIDE_C1, 0u);
  put<uint32_t>(kernarg, OFF_STRIDE_A0, (uint32_t)K);
  put<uint32_t>(kernarg, OFF_STRIDE_A1, 0u);
  put<uint32_t>(kernarg, OFF_STRIDE_B0, (uint32_t)K);
  put<uint32_t>(kernarg, OFF_STRIDE_B1, 0u);
  put<float>   (kernarg, OFF_ALPHA, 1.0f);
  put<float>   (kernarg, OFF_BETA,  0.0f);
  put<void*>   (kernarg, OFF_SCALE_A,  dScaleA);
  put<void*>   (kernarg, OFF_SCALE_B,  dScaleB);
  put<void*>   (kernarg, OFF_SCALE_C,  dScaleC);
  put<void*>   (kernarg, OFF_SCALE_D,  dScaleD);
  put<void*>   (kernarg, OFF_SCALE_AV, nullptr);
  put<void*>   (kernarg, OFF_BIAS,     nullptr);
  put<uint32_t>(kernarg, OFF_BIAS_TYPE,   0u);
  put<uint32_t>(kernarg, OFF_STRIDE_BIAS, 0u);
  put<float>   (kernarg, OFF_ACT_ALPHA, 0.0f);
  put<float>   (kernarg, OFF_ACT_BETA,  0.0f);
  put<uint32_t>(kernarg, OFF_ACT_TYPE, 0u);
  }
  // Always patch pointer fields with our allocations.
  // hipBLASLt convention with transA=T,transB=N:
  //   A operand pointer = N*K buffer (the "weights"), B operand = M*K buffer.
  put<void*>(kernarg, OFF_D, dD);
  put<void*>(kernarg, OFF_C, dD);
  put<void*>(kernarg, OFF_A, dB);  // dB holds N*K bytes
  put<void*>(kernarg, OFF_B, dA);  // dA holds M*K bytes
  put<void*>(kernarg, OFF_SCALE_A, dScaleA);
  put<void*>(kernarg, OFF_SCALE_B, dScaleB);
  put<void*>(kernarg, OFF_SCALE_C, dScaleC);
  put<void*>(kernarg, OFF_SCALE_D, dScaleD);
  put<void*>(kernarg, OFF_SCALE_AV, nullptr);
  put<void*>(kernarg, OFF_BIAS, nullptr);

  size_t arg_size = KERNARG_BYTES;
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE,    &arg_size,
                    HIP_LAUNCH_PARAM_END};

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  std::fprintf(stderr, "kernarg ready, launching warmup\n");
  // Warmup
  for (int i = 0; i < 5; ++i) {
    HIP_CHECK(hipExtModuleLaunchKernel(
        func,
        /*globalX=*/(uint32_t)N, /*globalY=*/(uint32_t)(M / 128), /*globalZ=*/1,
        /*localX=*/128, /*localY=*/1, /*localZ=*/1,
        /*sharedMem=*/0, /*stream=*/stream,
        nullptr, (void**)&config,
        nullptr, nullptr));
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  hipEvent_t e0, e1;
  HIP_CHECK(hipEventCreate(&e0));
  HIP_CHECK(hipEventCreate(&e1));
  HIP_CHECK(hipEventRecord(e0, stream));
  for (int i = 0; i < iters; ++i) {
    HIP_CHECK(hipExtModuleLaunchKernel(
        func, (uint32_t)N, (uint32_t)(M / 128), 1, 128, 1, 1,
        0, stream, nullptr, (void**)&config, nullptr, nullptr));
  }
  HIP_CHECK(hipEventRecord(e1, stream));
  HIP_CHECK(hipEventSynchronize(e1));
  float ms_total = 0;
  HIP_CHECK(hipEventElapsedTime(&ms_total, e0, e1));
  float ms = ms_total / iters;
  double tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
  std::printf("FP8 extracted: M=%d N=%d K=%d  %.4f ms  %.1f TFLOP/s  (iters=%d)\n",
              M, N, K, ms, tflops, iters);

  if (check) {
    HIP_CHECK(hipMemcpy(hOut.data(), dD, (size_t)M*N*sizeof(float),
                        hipMemcpyDeviceToHost));
    // cosine similarity
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < hOut.size(); ++i) {
      dot += (double)hOut[i] * hRef[i];
      na  += (double)hOut[i] * hOut[i];
      nb  += (double)hRef[i] * hRef[i];
    }
    double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    std::printf("  cos vs FP32 ref = %.6f\n", cos);
  }

  hipFree(dA); hipFree(dB); hipFree(dD);
  hipFree(dScaleA); hipFree(dScaleB); hipFree(dScaleC); hipFree(dScaleD);
  hipStreamDestroy(stream);
  hipModuleUnload(module);
  return 0;
}
