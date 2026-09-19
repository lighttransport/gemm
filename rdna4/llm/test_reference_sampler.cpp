/* Differential test against the actual pinned libllama CPU sampler. */
#include "reference_sampler.h"
#include "llama.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static llama_sampler *reference(const hllm_sampler_config &c, int n) {
    auto *s = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(s, llama_sampler_init_penalties(n, c.penalty_last_n,
        c.repetition, c.frequency, c.presence));
    if (c.temperature == 0) {
        llama_sampler_chain_add(s, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(s, llama_sampler_init_top_k(c.top_k));
        llama_sampler_chain_add(s, llama_sampler_init_top_p(c.top_p, 0));
        llama_sampler_chain_add(s, llama_sampler_init_min_p(c.min_p, 0));
        llama_sampler_chain_add(s, llama_sampler_init_temp(c.temperature));
        llama_sampler_chain_add(s, llama_sampler_init_dist(c.seed));
    }
    return s;
}

int main() {
    unsigned long checks = 0;
    for (int n : {1, 31, 129, 1025, 4097, 248320}) {
        for (uint32_t seed : {1u, 42u, 2026u}) {
            for (int mode = 0; mode < 8; ++mode) {
                hllm_sampler_config c;
                hllm_sampler_defaults(&c);
                c.seed = seed;
                if (mode == 1) c.top_k = 1;
                if (mode == 2) { c.top_k = 0; c.top_p = 1; c.min_p = .2f; }
                if (mode == 3) { c.top_k = 256; c.top_p = .8f; c.min_p = .05f; }
                if (mode == 4) { c.top_k = 0; c.top_p = .999f; }
                if (mode == 5) { c.repetition = 1.1f; c.frequency = .2f; c.presence = 1.5f; }
                if (mode == 6) { c.temperature = 0; c.repetition = 1.2f; }
                if (mode == 7) { c.top_k = 0; c.top_p = 1; c.temperature = 1; }
                auto *ours = hllm_sampler_create(&c, n);
                auto *ref = reference(c, n);
                if (!ours || !ref) return 2;
                std::mt19937 inputs(7319);
                std::vector<float> logits(n), got_l(n), got_p(n);
                std::vector<int> got_id(n);
                std::vector<llama_token_data> data(n);
                for (int i = 0; i < 70; ++i) {
                    int token = (int)(inputs() % n);
                    hllm_sampler_accept(ours, token);
                    llama_sampler_accept(ref, token);
                }
                const int steps = n > 4097 ? 8 : 160;
                for (int step = 0; step < steps; ++step) {
                    if (step == 50) { hllm_sampler_reset(ours); llama_sampler_reset(ref); }
                    for (int i = 0; i < n; ++i) {
                        // Ties and bucket boundaries exercise reference sorting.
                        logits[i] = step % 7 == 0 ? 0.0f : (int(inputs() % 4000) - 2000) * .01f;
                        data[i] = {i, logits[i], 0};
                    }
                    auto *clone = hllm_sampler_clone(ours);
                    int got = hllm_sampler_sample(ours, logits.data());
                    if (got != hllm_sampler_sample(clone, logits.data())) return 3;
                    hllm_sampler_free(clone);
                    llama_token_data_array a{data.data(), data.size(), -1, false};
                    llama_sampler_apply(ref, &a);
                    int want = data[a.selected].id;
                    if (got != want) {
                        fprintf(stderr, "token mismatch n=%d seed=%u mode=%d step=%d got=%d want=%d\n", n, seed, mode, step, got, want);
                        return 1;
                    }
                    if (c.temperature > 0) {
                        int count = hllm_sampler_candidates(ours, got_id.data(), got_l.data(), got_p.data(), n);
                        if (count != (int)a.size) return 4;
                        for (int i = 0; i < count; ++i) {
                            if (got_id[i] != data[i].id ||
                                memcmp(&got_l[i], &data[i].logit, sizeof(float)) ||
                                memcmp(&got_p[i], &data[i].p, sizeof(float))) {
                                fprintf(stderr, "candidate mismatch n=%d mode=%d step=%d index=%d\n", n, mode, step, i);
                                return 5;
                            }
                            ++checks;
                        }
                    }
                    hllm_sampler_accept(ours, got);
                    llama_sampler_accept(ref, want);
                    ++checks;
                }
                hllm_sampler_free(ours);
                llama_sampler_free(ref);
            }
        }
    }
    printf("PASS: %lu exact sampler comparisons (tokens, candidates, logits, probabilities, clone/reset)\n", checks);
}
