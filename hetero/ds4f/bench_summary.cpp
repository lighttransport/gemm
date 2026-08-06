// bench_summary.cpp -- DS4F prefill benchmark driver.
//
// Runs the test_hip_ds4f_real binary over the settled configurations and batch
// sizes, parses the "real hybrid prefill" lines, and prints a comparison table.
// The configs are the ones measured in 2026-08-06 (see README.md):
//   exact       -- dual dense/shared on ROCm, routed experts on CPU (0/64 gate)
//   rocm_expert -- HIP raw MXFP4 experts, 14 resident layers + streaming
//   cuda_stream -- all 43 layers' experts through the RTX cache (LRU streams)
//   split       -- ROCm head 14 layers + CUDA tail (dual)
//
// Build the binary with the EXACT flags (no -ffast-math) + the tile override:
//   make -C hetero/ds4f build/test_hip_ds4f_real \
//     CFLAGS="-O3 -Wall -Wextra -std=c11 -D_GNU_SOURCE -march=native -mavx2 \
//             -mfma -mf16c -ffp-contract=fast -DDS4F_MAX_MTILE=8192"
// Build:  g++ -O2 -std=c++17 -o bench_summary bench_summary.cpp
// Run:    ./bench_summary /tmp/ds4f_nocopy_ep8 [--binary ./build/test_hip_ds4f_real]
//
// The exact route stays 0/64; the accelerated routes are approximate (~1/64).
// The CUDA weight cache is one contiguous pool (default 12.5 GB -- a single
// cuMemAlloc reaches ~13 GB on this driver, vs ~10.8 GB for the per-tensor
// 4.46 MB allocations), so the split's 29 CUDA layers now preload fully.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

struct Point {
    int batch;
    double exact, rocm, cuda, split;
};

const char *kBinary = "./build/test_hip_ds4f_real";
const char *kStageDir = "/tmp/ds4f_nocopy_ep8";

// Base flags shared by every run.
std::string base_flags() {
    return std::string("--stage-dir ") + kStageDir +
           " --ep-size 8 --ep-rank 0 --layers 43 --bank-layers 43 --threads 16"
           " --cmgs 4 --prefill-context 0 --skip-cpu-ref 1 --cuda-device 0"
           " --hip-device 0 --hip-ordered-fp8-layers 43 --hip-fused-shared-ffn 1";
}

// The config-specific flags.  Key: exact / rocm / cuda / split.
std::string config_flags(const char *key) {
    if (!strcmp(key, "exact"))
        return "";
    if (!strcmp(key, "rocm"))
        return " --hip-mxfp4-resident-layers 14 --hip-mxfp4-stream-raw 1";
    if (!strcmp(key, "cuda"))
        return " --dual-gpu 1 --dual-cuda-small-buckets 1 --dual-cuda-terms 2"
               " --dual-cuda-resident-from 0";
    if (!strcmp(key, "split"))
        return " --dual-gpu 1 --dual-cuda-small-buckets 1 --dual-cuda-terms 2"
               " --dual-cuda-resident-from 14"
               " --hip-mxfp4-resident-layers 14 --hip-mxfp4-stream-raw 1"
               " --hip-prefill-attn 1";
    return "";
}

// Run the binary for one point; returns the parsed tok/s (0.0 on failure).
double run_point(const char *key, int batch, bool cuda) {
    std::string cmd = std::string(kBinary) + " " + base_flags() + " --dual-gpu 1"
                      + (cuda ? " --prefill-batch " : " --prefill-batch ")
                      + std::to_string(batch) + config_flags(key) + " 2>&1";
    FILE *p = popen(cmd.c_str(), "r");
    if (!p) return 0.0;
    char line[512];
    double tok = 0.0;
    while (fgets(line, sizeof(line), p)) {
        if (strstr(line, "real hybrid prefill:")) {
            const char *g = strstr(line, "gpu=");
            if (g) tok = atof(g + 4);
        }
    }
    pclose(p);
    return tok;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc > 1) kStageDir = argv[1];
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--binary") && i + 1 < argc) kBinary = argv[++i];
    }
    std::vector<int> batches = {256, 512, 1024, 2048, 4096};
    std::map<int, Point> rows;
    for (int b : batches) rows[b] = Point{b, 0, 0, 0, 0};

    for (auto &kv : rows) {
        int b = kv.first;
        Point &p = kv.second;
        p.exact = run_point("exact", b, true);
        p.rocm  = run_point("rocm",  b, true);
        p.cuda  = run_point("cuda",  b, true);
        p.split = run_point("split", b, true);
    }

    printf("DS4F prefill tok/s (approximate routes ~1/64; exact stays 0/64)\n");
    printf("%6s %10s %12s %12s %10s\n", "batch", "exact", "rocm-expert", "cuda-stream", "split");
    for (auto &kv : rows) {
        const Point &p = kv.second;
        printf("%6d %10.1f %12.1f %12.1f %10.1f\n",
               p.batch, p.exact, p.rocm, p.cuda, p.split);
    }
    return 0;
}
