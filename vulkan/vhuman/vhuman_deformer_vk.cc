#include "vhuman_deformer_vk.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../deps/vulkan-runner.hh"

using vl_cpp::vulkan::VulkanComputeRunner;

static const uint32_t kSpirv[] =
#include "vh_deform.spv.inc"
    ;
static const uint32_t kSpirvContacts[] =
#include "vh_contacts.spv.inc"
    ;

namespace {
constexpr int FT = 8, MAX_M = 256, MAX_J = 64;
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

struct vh_vk {
    vh_deformer *d = nullptr;
    VulkanComputeRunner runner;
    VulkanComputeRunner::ComputePipeline pipe{}, pipe_ct{};
    VulkanComputeRunner::BufferInfo c_ints{}, c_flts{}, c_scr{};
    int32_t ct_pc[26] = {};
    bool has_ct = false;
    std::vector<float> hscratch;
    VulkanComputeRunner::BufferInfo rest{}, morph{}, sj{}, sw{}, w{}, skin{}, out{}, stage{};
    size_t V = 0, M = 0, J = 0, C = 0, cap = 0;
    std::vector<float> hw, hskin;
    std::string name;
    bool have_pipe = false;
    bool bar_inputs = true, cached_readback = true;
};

static bool upload(vh_vk *g, VulkanComputeRunner::BufferInfo &b, const void *data, size_t bytes) {
    return g->runner.createDeviceLocalBuffer(bytes, b) && g->runner.uploadToDeviceLocal(b, data, bytes);
}

vh_vk *vh_vk_create(vh_deformer *d, int device, int verbose) {
    if (!vl_cpp::vulkan::InitializeVulkan()) return nullptr;
    vh_vk *g = new vh_vk();
    g->d = d;
    g->V = vh_deformer_vertices(d);
    g->M = vh_deformer_morphs(d);
    g->J = vh_deformer_joints(d);
    g->C = vh_deformer_controls(d);
    if (g->M > (size_t)MAX_M || g->J > (size_t)MAX_J || !g->runner.initialize(false) ||
        device >= (int)g->runner.getDeviceCount() || !g->runner.selectDevice((uint32_t)device)) {
        if (verbose) fprintf(stderr, "vhuman_deformer_vk: %s\n", g->runner.getLastError().c_str());
        delete g;
        return nullptr;
    }
    g->name = g->runner.getDeviceName((uint32_t)device);
    std::vector<uint32_t> spv(kSpirv, kSpirv + sizeof(kSpirv) / sizeof(kSpirv[0]));
    std::vector<VkDescriptorSetLayoutBinding> bindings(7);
    for (uint32_t i = 0; i < 7; ++i) {
        bindings[i] = {};
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    if (!g->runner.createComputePipelineWithPushConstants(spv, bindings, 16, g->pipe)) {
        if (verbose) fprintf(stderr, "vhuman_deformer_vk: %s\n", g->runner.getLastError().c_str());
        delete g;
        return nullptr;
    }
    g->have_pipe = true;
    const vh_contacts *c = vh_deformer_contacts(d);
    if (c && c->ns <= 512) {
        std::vector<int32_t> ints;
        std::vector<float> flts;
        auto put_i = [&](const int *p, size_t n) { int32_t o = (int32_t)ints.size(); ints.insert(ints.end(), p, p + n); return o; };
        auto put_f = [&](const float *p, size_t n) { int32_t o = (int32_t)flts.size(); flts.insert(flts.end(), p, p + n); return o; };
        std::vector<float> thr_t(c->nl * c->ns);                  // (T, L): coalesced
        for (size_t l = 0; l < c->nl; ++l)
            for (size_t t = 0; t < c->ns; ++t) thr_t[t * c->nl + l] = c->sph_thr[l * c->ns + t];
        int32_t *q = g->ct_pc;
        q[0] = (int32_t)g->V; q[1] = (int32_t)g->J; q[2] = (int32_t)c->ne; q[3] = (int32_t)c->nl;
        q[4] = (int32_t)c->ns; q[5] = (int32_t)c->np; q[6] = c->up_joints[0]; q[7] = c->up_joints[1];
        q[9] = put_i(c->eye_ids, c->ne); q[10] = put_i(c->eye_joint, c->ne); q[11] = put_i(c->lip_ids, c->nl);
        q[12] = put_i(c->sph_joint, c->ns * 2); q[13] = put_i(c->pair_u, c->np); q[14] = put_i(c->pair_l, c->np);
        q[15] = put_f(c->eye_center, c->ne * 3); q[16] = put_f(c->eye_thr, c->ne);
        q[17] = put_f(c->sph_center, c->ns * 3); q[18] = put_f(c->sph_weight, c->ns * 2);
        q[19] = put_f(thr_t.data(), thr_t.size()); q[20] = put_f(c->pair_floor, c->np);
        q[21] = (int32_t)c->nc; q[22] = VH_CONTACT_SMOOTH_STEPS;
        if (c->nc) {
            q[23] = put_i(c->verts, c->nc); q[24] = put_i(c->nbr_ptr, c->nc + 1);
            q[25] = put_i(c->nbr_idx, (size_t)c->nbr_ptr[c->nc]);
        }
        std::vector<uint32_t> spv2(kSpirvContacts, kSpirvContacts + sizeof(kSpirvContacts) / sizeof(kSpirvContacts[0]));
        std::vector<VkDescriptorSetLayoutBinding> b4(bindings.begin(), bindings.begin() + 5);
        g->has_ct = g->runner.createComputePipelineWithPushConstants(spv2, b4, sizeof(g->ct_pc), g->pipe_ct) &&
                    upload(g, g->c_ints, ints.data(), ints.size() * 4) && upload(g, g->c_flts, flts.data(), flts.size() * 4);
        if (!g->has_ct && verbose) fprintf(stderr, "vhuman_deformer_vk: no contact pipeline: %s\n", g->runner.getLastError().c_str());
    }
    size_t n3 = g->V * 3;
    if (!upload(g, g->rest, vh_deformer_rest(d), n3 * 4) || !upload(g, g->morph, vh_deformer_morph(d), g->M * n3 * 4) ||
        !upload(g, g->sj, vh_deformer_skin_joints(d), g->V * 16) || !upload(g, g->sw, vh_deformer_skin_weights(d), g->V * 16)) {
        vh_vk_free(g);
        return nullptr;
    }
    return g;
}

const char *vh_vk_name(const vh_vk *g) { return g->name.c_str(); }
int vh_vk_memory_flags(const vh_vk *g) { return (g->bar_inputs ? 1 : 0) | (g->cached_readback ? 2 : 0); }

static bool ensure(vh_vk *g, size_t frames) {
    if (frames <= g->cap) return true;
    auto &r = g->runner;
    for (auto *b : {&g->w, &g->skin, &g->out, &g->stage, &g->c_scr})
        if (b->buffer) { r.destroyBuffer(*b); *b = {}; }
    size_t cap = frames < 64 ? 64 : frames;
    VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkBufferUsageFlags use = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    // per-frame inputs: device-local + host-visible (resizable BAR) when available,
    // else plain host memory (then every workgroup reads them over the bus)
    VkMemoryPropertyFlags bar = host | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    auto input_buffer = [&](size_t bytes, VulkanComputeRunner::BufferInfo &b) {
        if (r.createBuffer(bytes, use, bar, b)) return true;
        g->bar_inputs = false;
        return r.createBuffer(bytes, use, host, b);
    };
    if (!input_buffer(cap * g->M * 4, g->w) || !input_buffer(cap * g->J * 48, g->skin) ||
        !r.createDeviceLocalBuffer(cap * g->V * 12, g->out) ||
        // readback: host-cached memory (uncached/write-combined reads are ~100x slower)
        !(r.createBuffer(cap * g->V * 12, VK_BUFFER_USAGE_TRANSFER_DST_BIT, host | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                         g->stage) ||
          ((g->cached_readback = false), r.createStagingBuffer(cap * g->V * 12, g->stage))))
        return false;
    g->hw.resize(cap * g->M);
    g->hskin.resize(cap * g->J * 12);
    g->hscratch.resize(vh_deformer_prepare_scratch(g->d, cap) + 1);
    g->cap = cap;
    if (g->has_ct && !(r.createDeviceLocalBuffer(cap * (size_t)(g->ct_pc[21] ? g->ct_pc[21] : 1) * 24, g->c_scr) &&
                       r.updateDescriptorSet(g->pipe_ct, {g->out, g->skin, g->c_ints, g->c_flts, g->c_scr})))
        return false;
    return r.updateDescriptorSet(g->pipe, {g->rest, g->morph, g->sj, g->sw, g->w, g->skin, g->out});
}

int vh_vk_eval_batch(vh_vk *g, const float *controls, size_t frames, int use_ml, float *out, double *ms4) {
    if (!frames) return 0;
    if (!ensure(g, frames)) return -1;
    auto &r = g->runner;
    double t0 = now_ms();
    vh_deformer_prepare_batch(g->d, controls, frames, use_ml, g->hw.data(), g->hskin.data(), g->hscratch.data());
    double t1 = now_ms();
    void *p = nullptr;
    if (!r.mapBuffer(g->w, &p)) return -2;
    memcpy(p, g->hw.data(), frames * g->M * 4);
    r.unmapBuffer(g->w);
    if (!r.mapBuffer(g->skin, &p)) return -2;
    memcpy(p, g->hskin.data(), frames * g->J * 48);
    r.unmapBuffer(g->skin);
    double t2 = now_ms();
    int32_t pc[4] = {(int32_t)g->V, (int32_t)g->M, (int32_t)g->J, (int32_t)frames};
    if (!r.beginRecording()) return -3;
    r.bindComputePipeline(g->pipe);
    r.bindDescriptorSets(g->pipe);
    r.pushConstants(g->pipe, pc, sizeof(pc));
    r.dispatch((uint32_t)((g->V + 127) / 128), (uint32_t)((frames + FT - 1) / FT), 1);
    int iters = vh_deformer_contact_iterations(g->d);
    if (g->has_ct && iters > 0) {                // exact contacts on the skinned result
        r.computeBarrier();
        g->ct_pc[8] = iters;
        r.bindComputePipeline(g->pipe_ct);
        r.bindDescriptorSets(g->pipe_ct);
        r.pushConstants(g->pipe_ct, g->ct_pc, sizeof(g->ct_pc));
        r.dispatch((uint32_t)frames, 1, 1);
    }
    if (!r.endRecordingAndSubmit() || !r.waitForCompletion()) return -4;
    double t3 = now_ms();
    if (out) {                                   // device -> host-cached staging -> out
        size_t bytes = frames * g->V * 12;
        if (!r.beginRecording()) return -5;
        r.computeToTransferBarrier();
        r.recordCopyBuffer(g->out, g->stage, bytes);
        if (!r.endRecordingAndSubmit() || !r.waitForCompletion()) return -5;
        if (!r.mapBuffer(g->stage, &p)) return -5;
        memcpy(out, p, bytes);
        r.unmapBuffer(g->stage);
    }
    double t4 = now_ms();
    if (ms4) {
        ms4[0] = t1 - t0;
        ms4[1] = t2 - t1;
        ms4[2] = t3 - t2;
        ms4[3] = t4 - t3;
    }
    return 0;
}

void vh_vk_free(vh_vk *g) {
    if (!g) return;
    auto &r = g->runner;
    for (auto *b : {&g->rest, &g->morph, &g->sj, &g->sw, &g->w, &g->skin, &g->out, &g->stage})
        if (b->buffer) r.destroyBuffer(*b);
    for (auto *b : {&g->c_ints, &g->c_flts, &g->c_scr})
        if (b->buffer) r.destroyBuffer(*b);
    if (g->has_ct) r.destroyComputePipeline(g->pipe_ct);
    if (g->have_pipe) r.destroyComputePipeline(g->pipe);
    r.cleanup();
    delete g;
}
