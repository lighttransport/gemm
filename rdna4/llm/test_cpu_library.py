"""Check the production CPU-library initializer and raw quantized dot kernels.

Usage: python3 test_cpu_library.py /path/to/libggml-cpu.so
No model or GPU is needed.
"""
from pathlib import Path
import subprocess
import sys
import tempfile

source = Path(__file__).with_name("hip_llm_runner.c").read_text()
start = source.index("static int hllm_init_cpu_library(")
end = source.index("static double hllm_monotonic_ms", start)
program = r'''
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
''' + source[start:end] + r'''
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 2; }
    if (hllm_init_cpu_library(lib) != 0) return 3;
    /* ggml_cpu_init is idempotent; multiple runners may load the library. */
    if (hllm_init_cpu_library(lib) != 0) return 3;
    void (*dot)(int, float *, size_t, const void *, size_t,
                const void *, size_t, int) = dlsym(lib, "ggml_vec_dot_q8_0_q8_0");
    void (*quant)(const float *, void *, int64_t) = dlsym(lib, "quantize_row_q8_0");
    if (!dot || !quant) return 4;
    float x[32], actual = 0;
    /* Exactly representable raw Q8_0 block: scale=1, all values=1. */
    struct { uint16_t scale; int8_t qs[32]; } block = {0x3c00, {0}};
    unsigned char xq[34];
    for (int i = 0; i < 32; ++i) { block.qs[i] = 1; x[i] = 1; }
    dot(32, &actual, 0, &block, 0, &block, 0, 1);
    if (fabsf(actual - 32) > 1e-5f) {
        fprintf(stderr, "raw dot: expected 32, got %g\n", actual);
        return 5;
    }
    quant(x, xq, 32);
    dot(32, &actual, 0, &block, 0, xq, 0, 1);
    if (fabsf(actual - 32) > .02f) return 6;
    dlclose(lib);
    return 0;
}
'''
if __name__ != "__main__":
    # unittest discovery imports every test_*.py file.  This is a standalone
    # command-line probe, so importing it must be side-effect free.
    pass
elif len(sys.argv) != 2:
    print(__doc__, file=sys.stderr)
    sys.exit(2)
else:
    with tempfile.TemporaryDirectory(prefix="qwen-cpu-library-") as directory:
        path = Path(directory)
        (path / "test.c").write_text(program)
        subprocess.run(["cc", "-Wall", "-Wextra", str(path / "test.c"),
                        "-ldl", "-lm", "-o", str(path / "test")], check=True)
        subprocess.run([str(path / "test"), sys.argv[1]], check=True)
    print("CPU library initialization and Q8 dot: PASS")
