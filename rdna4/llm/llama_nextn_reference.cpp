/* Independent dense NextN oracle. Link the pinned private llama.cpp build.
 * Inputs are emitted by the diagnostic LLM_QWEN35_MTP_TRACE runner option. */
#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s SIDECAR TRACE_PREFIX RECORDS\n", argv[0]);
        return 2;
    }
    const int records = atoi(argv[3]);
    if (records < 1 || records > 3) return 2;
    llama_backend_init();
    auto *device = ggml_backend_dev_by_name("ROCm0");
    if (!device) return 3;
    ggml_backend_dev_t devices[] = {device, nullptr};
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.load_mtp = true;
    mp.devices = devices;
    auto *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 4;
    auto cp = llama_context_default_params();
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cp.n_ctx = 8192;
    cp.n_batch = cp.n_ubatch = 1;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto *ctx = llama_init_from_model(model, cp);
    if (!ctx) return 5;
    int ne = llama_model_n_embd(model);
    int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> hidden(ne), ours(nv);
    bool pass = true;
    for (int i = 0; i < records; ++i) {
        std::string base = std::string(argv[2])+"."+std::to_string(i);
        FILE *f = fopen((base+".input").c_str(), "rb");
        int32_t header[4];
        if (!f || fread(header, sizeof(header), 1, f) != 1 ||
            header[0] != ne || header[1] != nv ||
            fread(hidden.data(), sizeof(float), ne, f) != (size_t)ne) return 6;
        if (fclose(f)) return 6;
        f = fopen((base+".logits").c_str(), "rb");
        if (!f || fread(ours.data(), sizeof(float), nv, f) != (size_t)nv) return 6;
        if (fclose(f)) return 6;
        llama_token token = header[2];
        llama_pos pos = header[3];
        if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, pos, -1)) return 7;
        int32_t nseq = 1;
        llama_seq_id seq = 0, *seqp = &seq;
        int8_t output = 1;
        llama_batch batch{};
        batch.n_tokens = 1;
        batch.token = &token;
        batch.embd = hidden.data();
        batch.pos = &pos;
        batch.n_seq_id = &nseq;
        batch.seq_id = &seqp;
        batch.logits = &output;
        if (llama_decode(ctx, batch)) return 8;
        auto *ref = llama_get_logits_ith(ctx, -1);
        if (!ref) return 8;
        double err = 0, norm = 0, maxerr = 0;
        bool finite = true;
        for (int j = 0; j < nv; ++j) {
            double d = double(ours[j])-ref[j];
            finite &= std::isfinite(d);
            err += d*d; norm += double(ref[j])*ref[j];
            maxerr = std::max(maxerr, std::abs(d));
        }
        int a = std::max_element(ours.begin(), ours.end())-ours.begin();
        int b = std::max_element(ref, ref+nv)-ref;
        double rel = std::sqrt(err/std::max(norm, 1e-30));
        bool ok = finite && a == b && rel < .05;
        pass &= ok;
        printf("NEXTN record=%d pos=%d input=%d ours=%d reference=%d relative_l2=%.9g max_abs=%.9g %s\n",
            i, pos, token, a, b, rel, maxerr, ok ? "PASS" : "FAIL");
        f = fopen((base+".reference.logits").c_str(), "wb");
        if (!f || fwrite(ref, sizeof(float), nv, f) != (size_t)nv) return 9;
        if (fclose(f)) return 9;
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return pass ? 0 : 1;
}
