// Sampling arithmetic and candidate sorting ported from llama.cpp
// 1859b520910af6f682256fd7299797774111a27a (src/llama-sampler.cpp).
/*
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "reference_sampler.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <deque>
#include <random>
#include <vector>
#include <new>

struct candidate { int32_t id; float logit, p; };
struct candidate_array { candidate *data; size_t size; bool sorted; };
struct hllm_sampler {
    hllm_sampler_config config;
    std::mt19937 rng;
    std::deque<int> history;
    std::vector<int> counts;
    std::vector<candidate> candidates;
    size_t size = 0;
    hllm_sampler(const hllm_sampler_config &c, int n)
        : config(c), rng(c.seed), counts(n, 0), candidates(n) {}
};

// writes result in res, does not mutate cur
static void candidate_array_partial_sort(const candidate_array & cur, int npartial, std::vector<candidate> & res) {
    static const auto comp = [](const candidate & a, const candidate & b) {
        return a.logit > b.logit;
    };

    constexpr int   nbuckets     = 128;
    constexpr float bucket_low   = -10.0f;
    constexpr float bucket_high  =  10.0f;
    constexpr float bucket_scale = nbuckets/(bucket_high - bucket_low);
    constexpr float bucket_inter = -bucket_low * bucket_scale;

    std::vector<int> bucket_idx;
    std::vector<int> histo(nbuckets, 0);

    std::vector<candidate*> bucket_ptrs;

    bucket_idx.reserve(cur.size);

    for (int i = 0; i < (int)cur.size; ++i) {
        const float val = cur.data[i].logit;
        // Preserve the reference buckets while avoiding an undefined float-to-
        // int conversion for masked (-infinity) or very large finite logits.
        int ib = val <= bucket_low ? 0 : val >= bucket_high ? nbuckets - 1 :
                 int(bucket_scale * val + bucket_inter);
        ib = std::max(0, std::min(nbuckets - 1, ib));
        bucket_idx.push_back(ib);
        ++histo[ib];
    }
    int nhave = 0;
    int ib = nbuckets - 1;
    for ( ; ib >= 0; --ib) {
        nhave += histo[ib];
        if (nhave >= npartial) {
            break;
        }
    }
    res.resize(nhave);
    auto * ptr = res.data();
    bucket_ptrs.reserve(nbuckets - ib);
    for (int j = nbuckets - 1; j >= ib; --j) {
        bucket_ptrs.push_back(ptr);
        ptr += histo[j];
    }
    for (int i = 0; i < (int)cur.size; ++i) {
        int j = bucket_idx[i];
        if (j >= ib) {
            *bucket_ptrs[nbuckets - 1 - j]++ = cur.data[i];
        }
    }

    ptr = res.data();
    int ndone = 0;
    for (int j = nbuckets - 1; j > ib; --j) {
        std::sort(ptr, ptr + histo[j], comp);
        ptr += histo[j];
        ndone += histo[j];
    }
    std::partial_sort(ptr, ptr + npartial - ndone, ptr + histo[ib], comp);
}

// reduces the size of cur_p to npartial, keeping only the top npartial elements
static void candidate_array_partial_sort_inplace(candidate_array * cur_p, int npartial) {
    static const auto comp = [](const candidate & a, const candidate & b) {
        return a.logit > b.logit;
    };

    if (npartial <= 128) {
        std::partial_sort(cur_p->data, cur_p->data + npartial, cur_p->data + cur_p->size, comp);

        cur_p->size = npartial;
        cur_p->sorted = true;

        return;
    }

    std::vector<candidate> tmp;

    candidate_array_partial_sort(*cur_p, npartial, tmp);

    std::copy(tmp.data(), tmp.data() + npartial, cur_p->data);

    cur_p->size = npartial;
    cur_p->sorted = true;
}

extern "C" void hllm_sampler_defaults(hllm_sampler_config *c) {
    if (c) *c = {42, 20, 0.95f, 0.0f, 0.6f, 64, 1.0f, 0.0f, 0.0f};
}

extern "C" hllm_sampler *hllm_sampler_create(const hllm_sampler_config *c, int n) {
    if (!c || n <= 0 || c->penalty_last_n < 0 ||
        !std::isfinite(c->top_p) || c->top_p < 0 || c->top_p > 1 ||
        !std::isfinite(c->min_p) || c->min_p < 0 || c->min_p > 1 ||
        !std::isfinite(c->temperature) || c->temperature < 0 ||
        !std::isfinite(c->repetition) || c->repetition <= 0 ||
        !std::isfinite(c->presence) || !std::isfinite(c->frequency)) return nullptr;
    try { return new hllm_sampler(*c, n); } catch (...) { return nullptr; }
}

extern "C" hllm_sampler *hllm_sampler_clone(const hllm_sampler *s) {
    if (!s) return nullptr;
    try { return new hllm_sampler(*s); } catch (...) { return nullptr; }
}

extern "C" void hllm_sampler_free(hllm_sampler *s) { delete s; }

extern "C" void hllm_sampler_reset(hllm_sampler *s) {
    if (!s) return;
    s->rng.seed(s->config.seed);
    s->history.clear();
    std::fill(s->counts.begin(), s->counts.end(), 0);
    s->size = 0;
}

extern "C" int hllm_sampler_accept(hllm_sampler *s, int token) {
    if (!s || token < 0 || token >= (int)s->counts.size()) return -1;
    if (s->config.penalty_last_n == 0) return 0;
    try {
        s->history.push_back(token);
        ++s->counts[token];
        if (s->history.size() > (size_t)s->config.penalty_last_n) {
            --s->counts[s->history.front()];
            s->history.pop_front();
        }
    } catch (...) { return -1; }
    return 0;
}

static void softmax(candidate_array &a) {
    float max_l = a.data[0].logit;
    if (!a.sorted)
        for (size_t i = 1; i < a.size; ++i) max_l = std::max(max_l, a.data[i].logit);
    float sum = 0.0f;
    for (size_t i = 0; i < a.size; ++i) {
        a.data[i].p = expf(a.data[i].logit - max_l);
        sum += a.data[i].p;
    }
    for (size_t i = 0; i < a.size; ++i) a.data[i].p /= sum;
}

static int sample(hllm_sampler &s, const float *logits) {
    const auto &c = s.config;
    candidate_array a{s.candidates.data(), s.candidates.size(), false};
    bool any_finite = false;
    for (size_t i = 0; i < a.size; ++i) {
        float value = logits[i];
        // A corrupt forward must fail, never silently generate token zero.
        if (std::isnan(value) || value == INFINITY) return -1;
        any_finite |= std::isfinite(value);
        if (s.counts[i] > 0) {
            value = value <= 0 ? value * c.repetition : value / c.repetition;
            value -= float(s.counts[i]) * c.frequency + c.presence;
        }
        a.data[i] = {(int32_t)i, value, 0.0f};
    }
    if (!any_finite) return -1;
    if (c.temperature == 0.0f) {
        size_t best = 0;
        for (size_t i = 1; i < a.size; ++i)
            if (a.data[i].logit > a.data[best].logit) best = i;
        s.size = a.size;
        return a.data[best].id;
    }
    if (c.top_k > 0)
        candidate_array_partial_sort_inplace(&a, std::min(c.top_k, (int)a.size));
    if (c.top_p < 1.0f) {
        softmax(a);
        std::vector<candidate> buffer;
        size_t k = a.size;
        candidate *data = a.data;
        if (!a.sorted && a.size > 1024) {
            k = std::min<size_t>(256, a.size);
            candidate_array_partial_sort(a, (int)k, buffer);
            data = buffer.data();
        } else if (!a.sorted) {
            candidate_array_partial_sort_inplace(&a, (int)k);
        }
        float sum = 0.0f;
        size_t last = a.size;
        for (size_t i = 0; i < a.size; ++i) {
            sum += data[i].p;
            if (sum >= c.top_p) { last = i + 1; break; }
            if (!a.sorted && i == k - 1) {
                k = a.size;
                candidate_array_partial_sort(a, (int)k, buffer);
                data = buffer.data();
            }
        }
        if (!a.sorted) {
            std::copy(buffer.data(), buffer.data() + last, a.data);
            a.sorted = true;
        }
        a.size = last;
    }
    if (c.min_p > 0.0f) {
        if (!a.sorted) {
            float max_l = -FLT_MAX;
            for (size_t i = 0; i < a.size; ++i) max_l = std::max(max_l, a.data[i].logit);
            float threshold = max_l + logf(c.min_p);
            size_t n = 0;
            for (size_t i = 0; i < a.size; ++i)
                if (a.data[i].logit >= threshold) a.data[n++] = a.data[i];
            a.size = n;
        } else {
            const float threshold = a.data[0].logit + logf(c.min_p);
            size_t i = 1;
            while (i < a.size && a.data[i].logit >= threshold) ++i;
            a.size = i;
        }
    }
    if (a.size == 0) return -1;
    for (size_t i = 0; i < a.size; ++i) a.data[i].logit /= c.temperature;
    s.size = a.size;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    if (a.size == 1) {
        distribution(s.rng);
        a.data[0].p = 1.0f;
        return a.data[0].id;
    }
    float max_l = a.data[0].logit;
    if (!a.sorted)
        for (size_t i = 1; i < a.size; ++i) max_l = std::max(max_l, a.data[i].logit);
    double sum = 0.0;
    for (size_t i = 0; i < a.size; ++i) {
        a.data[i].p = expf(a.data[i].logit - max_l);
        sum += a.data[i].p;
    }
    const double target = sum * distribution(s.rng);
    double running = 0.0;
    int selected = -1;
    for (size_t i = 0; i < a.size; ++i) {
        if (selected < 0) {
            running += a.data[i].p;
            if (running >= target) selected = a.data[i].id;
        }
        a.data[i].p /= sum;
    }
    return selected;
}

extern "C" int hllm_sampler_sample(hllm_sampler *s, const float *logits) {
    if (!s || !logits) return -1;
    s->size = 0;
    try { return sample(*s, logits); } catch (...) { return -1; }
}

extern "C" int hllm_sampler_candidates(const hllm_sampler *s, int *ids,
                                       float *logits, float *p, int capacity) {
    if (!s || capacity < (int)s->size) return -1;
    for (size_t i = 0; i < s->size; ++i) {
        if (ids) ids[i] = s->candidates[i].id;
        if (logits) logits[i] = s->candidates[i].logit;
        if (p) p[i] = s->candidates[i].p;
    }
    return (int)s->size;
}
