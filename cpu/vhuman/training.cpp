/* CPU training primitives and two-layer GRU BPTT using repository GEMM. */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef VH_TRAIN_AVX2
#include "../../ryzen/gemm_avx2.h"
#endif

static thread_local std::string train_error;
static thread_local int training_threads = 1;
static void require(bool condition, const char *message)
{
    if (!condition) throw std::invalid_argument(message);
}
static void finite(const float *x, size_t n)
{
    require(x != nullptr, "missing tensor");
    for (size_t i = 0; i < n; ++i) require(std::isfinite(x[i]), "nonfinite tensor");
}
static float sigmoid(float x)
{
    return x >= 0 ? 1.f / (1.f + std::exp(-x)) : std::exp(x) / (1.f + std::exp(x));
}
static void gemm(float *out, const float *a, const float *b, int m, int n, int k,
                 bool transpose_a = false, bool transpose_b = false)
{
    std::vector<float> at, bt;
    if (transpose_a) {
        at.resize(size_t(m) * k);
        for (int i = 0; i < m; ++i) for (int j = 0; j < k; ++j) at[size_t(i)*k+j] = a[size_t(j)*m+i];
        a = at.data();
    }
    if (transpose_b) {
        bt.resize(size_t(k) * n);
        for (int i = 0; i < k; ++i) for (int j = 0; j < n; ++j) bt[size_t(i)*n+j] = b[size_t(j)*k+i];
        b = bt.data();
    }
#ifdef VH_TRAIN_AVX2
    int jobs = m >= 128 && size_t(m)*n*k >= 1048576 ? training_threads : 1;
#pragma omp parallel for num_threads(jobs) if(jobs > 1)
    for (int job = 0; job < jobs; ++job) {
        int first = int(int64_t(m)*job/jobs), last = int(int64_t(m)*(job+1)/jobs);
        if (last > first) sgemm_avx2(last-first,n,k,1.f,a+size_t(first)*k,k,b,n,0.f,out+size_t(first)*n,n);
    }
#else
    for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) {
        float sum = 0;
        for (int p = 0; p < k; ++p) sum += a[size_t(i)*k+p] * b[size_t(p)*n+j];
        out[size_t(i)*n+j] = sum;
    }
#endif
}
static void adamw(float *p, const float *g, float *m, float *v, size_t count,
                  int step, double lr, double decay, double clip)
{
    require(count && step > 0 && std::isfinite(lr) && lr >= 0 && std::isfinite(decay) && decay >= 0 &&
            std::isfinite(clip) && clip >= 0 && lr*decay <= 1, "invalid optimizer arguments");
    finite(p,count); finite(g,count); finite(m,count); finite(v,count);
    double norm2 = 0;
    for (size_t i = 0; i < count; ++i) {
        require(v[i] >= 0, "negative second moment");
        norm2 += double(g[i])*g[i];
    }
    double scale = clip > 0 ? std::min(1., clip/(std::sqrt(norm2)+1e-6)) : 1.;
    double correction1 = 1-std::pow(.9,step), correction2 = std::sqrt(1-std::pow(.999,step));
    for (size_t i = 0; i < count; ++i) {
        float gradient = float(g[i]*scale);
        m[i] = .9f*m[i] + .1f*gradient;
        v[i] = .999f*v[i] + .001f*gradient*gradient;
        p[i] = float(p[i]*(1-lr*decay) - lr/correction1*m[i]/(std::sqrt(double(v[i]))/correction2+1e-8));
    }
}

struct motion_trainer {
    static constexpr int R = 128, G = 384;
    int hidden, controls, output, steps = 0;
    size_t offsets[14]{};
    std::vector<float> parameters, gradients, moment1, moment2;
    enum { PW, PB, EMB, IW0, HW0, IB0, HB0, IW1, HW1, IB1, HB1, OW, OB };
    motion_trainer(int h, int c): hidden(h), controls(c), output(8*c)
    {
        require(h >= 1 && h <= 16384 && c >= 1 && c <= 512, "invalid motion dimensions");
        size_t lengths[] = {size_t(R)*h, R, 2048*16, G*G, G*R, G, G, G*R, G*R, G, G, size_t(output)*R, size_t(output)};
        for (int i = 0; i < 13; ++i) offsets[i+1] = offsets[i]+lengths[i];
        parameters.resize(offsets[13]); gradients.resize(offsets[13]);
        moment1.resize(offsets[13]); moment2.resize(offsets[13]);
    }
    float *p(int index) { return parameters.data()+offsets[index]; }
    float *g(int index) { return gradients.data()+offsets[index]; }
    void bias_gradient(int index, const std::vector<float> &value, int length, int t)
    {
        for (int i = 0; i < t; ++i) for (int j = 0; j < length; ++j) g(index)[j] += value[size_t(i)*length+j];
    }
    double compute(const float *hidden_input, const int32_t *codes, const float *target,
                   const float *bounds, const float *weights, float *state, float *prediction,
                   int t, int mode, double lr)
    {
        require(t > 0 && t <= 32 && mode >= 0 && mode <= 2 && codes && state && prediction, "invalid motion batch");
        finite(hidden_input,size_t(t)*hidden); finite(target,size_t(t)*output);
        finite(bounds,size_t(controls)*2); finite(weights,controls); finite(state,2*R);
        finite(parameters.data(),parameters.size());
        for (int c = 0; c < controls; ++c) require(bounds[2*c] <= bounds[2*c+1] && weights[c] >= 0, "invalid ranges/loss weights");
        for (int i = 0; i < t*16; ++i) require(codes[i] >= 0 && codes[i] < 2048, "invalid code index");
        if (mode == 2) require(std::isfinite(lr) && lr > 0 && lr <= 1, "invalid learning rate");
        std::vector<float> projected(size_t(t)*R), input(size_t(t)*G), logits(size_t(t)*output);
        gemm(projected.data(),hidden_input,p(PW),t,R,hidden,false,true);
        for (int i = 0; i < t; ++i) {
            for (int j = 0; j < R; ++j) input[size_t(i)*G+j] = projected[size_t(i)*R+j]+p(PB)[j];
            for (int j = 0; j < 16; ++j) std::memcpy(input.data()+size_t(i)*G+R+16*j,p(EMB)+16*codes[i*16+j],16*sizeof(float));
        }
        struct layer_cache {
            std::vector<float> h, previous, reset, update, candidate, hidden_candidate;
            explicit layer_cache(int t): h(size_t(t)*R), previous(h.size()), reset(h.size()),
                update(h.size()), candidate(h.size()), hidden_candidate(h.size()) {}
        };
        layer_cache layers[] = {layer_cache(t),layer_cache(t)};
        float gi[G], gh[G];
        for (int i = 0; i < t; ++i) for (int l = 0; l < 2; ++l) {
            auto &cache = layers[l];
            const float *previous = i ? cache.h.data()+size_t(i-1)*R : state+l*R;
            std::memcpy(cache.previous.data()+size_t(i)*R,previous,R*sizeof(float));
            const float *x = l ? layers[0].h.data()+size_t(i)*R : input.data()+size_t(i)*G;
            gemm(gi,x,p(l ? IW1 : IW0),1,G,l ? R : G,false,true);
            gemm(gh,previous,p(l ? HW1 : HW0),1,G,R,false,true);
            for (int j = 0; j < G; ++j) { gi[j] += p(l ? IB1 : IB0)[j]; gh[j] += p(l ? HB1 : HB0)[j]; }
            finite(gi,G); finite(gh,G);
            for (int j = 0; j < R; ++j) {
                size_t q = size_t(i)*R+j;
                float r = sigmoid(gi[j]+gh[j]), z = sigmoid(gi[R+j]+gh[R+j]);
                float n = std::tanh(gi[2*R+j]+r*gh[2*R+j]);
                cache.reset[q]=r; cache.update[q]=z; cache.candidate[q]=n; cache.hidden_candidate[q]=gh[2*R+j];
                cache.h[q]=(1-z)*n+z*previous[j];
            }
        }
        gemm(logits.data(),layers[1].h.data(),p(OW),t,output,R,false,true);
        finite(logits.data(),logits.size());
        std::vector<float> dy(size_t(t)*output), dlogit(dy.size());
        double loss = 0;
        for (int i = 0; i < t*output; ++i) {
            int c = i%controls;
            float prob = sigmoid(logits[i]+p(OB)[i%output]);
            prediction[i] = bounds[2*c]+prob*(bounds[2*c+1]-bounds[2*c]);
            float difference = prediction[i]-target[i], a = std::abs(difference);
            loss += weights[c]*(a < .1f ? .5*double(difference)*difference/.1 : a-.05)/(t*output);
            dy[i] = weights[c]*std::clamp(difference/.1f,-1.f,1.f)/(t*output);
            dlogit[i] = prob*(1-prob)*(bounds[2*c+1]-bounds[2*c]);
        }
        int temporal_count = (t*8-1)*controls;
        for (int i = controls; i < t*output; ++i) {
            float delta = prediction[i]-prediction[i-controls]-target[i]+target[i-controls];
            float factor = float(.05*weights[i%controls]/temporal_count);
            loss += factor*std::abs(delta);
            float derivative = factor*((delta > 0)-(delta < 0));
            dy[i] += derivative; dy[i-controls] -= derivative;
        }
        require(std::isfinite(loss), "nonfinite motion loss");
        finite(prediction,size_t(t)*output);
        finite(layers[0].h.data(),size_t(t)*R); finite(layers[1].h.data(),size_t(t)*R);
        if (mode) {
            std::fill(gradients.begin(),gradients.end(),0.f);
            for (size_t i = 0; i < dy.size(); ++i) dlogit[i] *= dy[i];
            gemm(g(OW),dlogit.data(),layers[1].h.data(),output,R,t,true,false);
            bias_gradient(OB,dlogit,output,t);
            std::vector<float> dh[2] = {std::vector<float>(size_t(t)*R),std::vector<float>(size_t(t)*R)};
            gemm(dh[1].data(),dlogit.data(),p(OW),t,R,output);
            std::vector<float> dgi[2] = {std::vector<float>(size_t(t)*G),std::vector<float>(size_t(t)*G)};
            std::vector<float> dgh[2] = {std::vector<float>(size_t(t)*G),std::vector<float>(size_t(t)*G)};
            std::vector<float> dx(size_t(t)*G);
            float recurrent[2][R] = {}, next[R], du[G];
            for (int i = t-1; i >= 0; --i) for (int l = 1; l >= 0; --l) {
                auto &cache = layers[l];
                float *ig = dgi[l].data()+size_t(i)*G, *hg = dgh[l].data()+size_t(i)*G;
                for (int j = 0; j < R; ++j) {
                    size_t q = size_t(i)*R+j;
                    float d = dh[l][q]+recurrent[l][j], r = cache.reset[q], z = cache.update[q], n = cache.candidate[q];
                    float dn = d*(1-z)*(1-n*n), dz = d*(cache.previous[q]-n)*z*(1-z);
                    float dr = dn*cache.hidden_candidate[q]*r*(1-r);
                    ig[j]=dr; ig[R+j]=dz; ig[2*R+j]=dn;
                    hg[j]=dr; hg[R+j]=dz; hg[2*R+j]=dn*r;
                    recurrent[l][j]=d*z;
                }
                gemm(next,hg,p(l ? HW1 : HW0),1,R,G);
                for (int j = 0; j < R; ++j) recurrent[l][j] += next[j];
                gemm(du,ig,p(l ? IW1 : IW0),1,l ? R : G,G);
                if (l) for (int j = 0; j < R; ++j) dh[0][size_t(i)*R+j] += du[j];
                else std::memcpy(dx.data()+size_t(i)*G,du,G*sizeof(float));
            }
            for (int l = 0; l < 2; ++l) {
                gemm(g(l ? IW1 : IW0),dgi[l].data(),l ? layers[0].h.data() : input.data(),G,l ? R : G,t,true,false);
                gemm(g(l ? HW1 : HW0),dgh[l].data(),layers[l].previous.data(),G,R,t,true,false);
                bias_gradient(l ? IB1 : IB0,dgi[l],G,t); bias_gradient(l ? HB1 : HB0,dgh[l],G,t);
            }
            std::vector<float> dp(size_t(t)*R);
            for (int i = 0; i < t; ++i) {
                std::memcpy(dp.data()+size_t(i)*R,dx.data()+size_t(i)*G,R*sizeof(float));
                for (int j = 0; j < 16; ++j) for (int k = 0; k < 16; ++k)
                    g(EMB)[16*codes[i*16+j]+k] += dx[size_t(i)*G+R+16*j+k];
            }
            gemm(g(PW),dp.data(),hidden_input,R,hidden,t,true,false); bias_gradient(PB,dp,R,t);
            finite(gradients.data(),gradients.size());
            if (mode == 2) adamw(parameters.data(),gradients.data(),moment1.data(),moment2.data(),parameters.size(),++steps,lr,.01,1.);
        }
        for (int l = 0; l < 2; ++l) std::memcpy(state+l*R,layers[l].h.data()+size_t(t-1)*R,R*sizeof(float));
        return loss;
    }
};

extern "C" {
const char *vh_train_error() { return train_error.c_str(); }
int vh_train_set_threads(int threads)
{
    if (threads < 1 || threads > 64) { train_error="threads must be 1..64"; return -1; }
    training_threads=threads;
    return 0;
}
int vh_train_gemm(float *out, const float *a, const float *b, int m, int n, int k, int ta, int tb)
{
    try {
        require(out && m > 0 && n > 0 && k > 0 && ta >= 0 && ta <= 1 && tb >= 0 && tb <= 1, "invalid GEMM");
        finite(a,size_t(m)*k); finite(b,size_t(k)*n); gemm(out,a,b,m,n,k,ta,tb); finite(out,size_t(m)*n);
        return 0;
    } catch (const std::exception &e) { train_error=e.what(); return -1; }
}
int vh_train_adamw(float *p, const float *g, float *m, float *v, size_t n, int step, double lr, double decay, double clip)
{
    try { adamw(p,g,m,v,n,step,lr,decay,clip); return 0; }
    catch (const std::exception &e) { train_error=e.what(); return -1; }
}
void *vh_train_motion_open(int h, int c)
{
    try { return new motion_trainer(h,c); }
    catch (const std::exception &e) { train_error=e.what(); return nullptr; }
}
void vh_train_motion_close(void *handle) { delete static_cast<motion_trainer *>(handle); }
float *vh_train_motion_data(void *handle, int kind)
{
    auto *m = static_cast<motion_trainer *>(handle);
    return m && (kind == 0 || kind == 1) ? (kind ? m->gradients.data() : m->parameters.data()) : nullptr;
}
size_t vh_train_motion_size(void *handle)
{
    auto *m = static_cast<motion_trainer *>(handle); return m ? m->parameters.size() : 0;
}
int vh_train_motion_compute(void *handle, const float *h, const int32_t *codes, const float *target,
                            const float *bounds, const float *weights, float *state, float *prediction,
                            int t, int mode, double lr, double *loss)
{
    try {
        require(handle && loss, "missing motion trainer");
        *loss=static_cast<motion_trainer *>(handle)->compute(h,codes,target,bounds,weights,state,prediction,t,mode,lr);
        return 0;
    } catch (const std::exception &e) { train_error=e.what(); return -1; }
}
}
