"""Compile the production sampler and check softmax shift invariance."""
from pathlib import Path
import subprocess
import tempfile

source = Path(__file__).with_name("test_hip_llm.c").read_text()
start = source.index("static int sample_top_k_p(")
end = source.index("static int argmax_logits", start)
program = "#include <math.h>\n#include <stdint.h>\n#include <stddef.h>\n" + source[start:end] + r'''
int main(void) {
    const float a[] = {0, -1, -2, -3};
    const float b[] = {20, 19, 18, 17};
    unsigned ra = 1234, rb = 1234;
    int counts[4] = {0};
    for (int i = 0; i < 20000; ++i) {
        int x = sample_top_k_p(a, 4, 4, 1, 1, 0, 0,
                               NULL, NULL, NULL, 0, 0, &ra);
        int y = sample_top_k_p(b, 4, 4, 1, 1, 0, 0,
                               NULL, NULL, NULL, 0, 0, &rb);
        if (x != y) return 1;
        counts[x]++;
    }
    /* Expected softmax probability of the first token is approximately .644. */
    return counts[0] < 12000 || counts[0] > 13800;
}
'''
with tempfile.TemporaryDirectory(prefix="qwen-sampler-") as directory:
    path = Path(directory)
    (path / "test.c").write_text(program)
    subprocess.run(["cc", "-Wall", "-Wextra", str(path / "test.c"),
                    "-lm", "-o", str(path / "test")], check=True)
    subprocess.run([str(path / "test")], check=True)
print("Sampler shift invariance and distribution: PASS")
