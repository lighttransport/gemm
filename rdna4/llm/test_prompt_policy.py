"""CPU-only regression for the runner's GGUF BOS insertion policy."""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
source = Path(__file__).with_name("test_hip_llm.c").read_text()
start = source.index("static int prompt_bos_id(")
end = source.index("static int run_stdio_server(", start)
program = r'''
#include <assert.h>
#include <stdlib.h>
#define GGUF_LOADER_IMPLEMENTATION
#include "common/gguf_loader.h"
''' + source[start:end] + r'''
int main(void) {
    gguf_kv kv[2] = {0};
    gguf_context ctx = {0};
    ctx.kv = kv;
    ctx.n_kv = 2;
    kv[0].key.str = "tokenizer.ggml.bos_token_id";
    kv[0].type = GGUF_TYPE_UINT32;
    kv[0].value.u32 = 248044;
    kv[1].key.str = "tokenizer.ggml.add_bos_token";
    kv[1].type = GGUF_TYPE_BOOL;
    unsetenv("LLM_ADD_BOS");
    assert(prompt_bos_id(&ctx) == -1);
    kv[1].value.b = 1;
    assert(prompt_bos_id(&ctx) == 248044);
    ctx.n_kv = 1;
    assert(prompt_bos_id(&ctx) == 248044);
    ctx.n_kv = 0;
    assert(prompt_bos_id(&ctx) == -1);
    setenv("LLM_ADD_BOS", "42", 1);
    assert(prompt_bos_id(&ctx) == 42);
    setenv("LLM_ADD_BOS", "0", 1);
    assert(prompt_bos_id(&ctx) == 0);
    assert(gguf_type_name(GGUF_TYPE_BOOL) != NULL);
    assert(ggml_type_name(0) != NULL);
    return 0;
}
'''
with tempfile.TemporaryDirectory(prefix="qwen-prompt-policy-") as directory:
    path = Path(directory)
    (path / "test.c").write_text(program)
    subprocess.run(["cc", "-Wall", "-Wextra", "-I", str(root),
                    str(path / "test.c"), "-o", str(path / "test")], check=True)
    subprocess.run([str(path / "test")], check=True)
print("GGUF BOS policy: PASS")
