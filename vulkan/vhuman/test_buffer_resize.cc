// CPU-only allocation/descriptor fault injection for the production ensure().
// test_native_safety.py supplies that function verbatim as an include.
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <new>
#include <stdexcept>
#include <vector>

using VkMemoryPropertyFlags = unsigned;
using VkBufferUsageFlags = unsigned;
constexpr unsigned VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 1, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 2,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 4, VK_MEMORY_PROPERTY_HOST_CACHED_BIT = 8,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 1, VK_BUFFER_USAGE_TRANSFER_DST_BIT = 2;

struct VulkanComputeRunner {
    struct BufferInfo { uintptr_t buffer = 0; };
    int calls = 0, fail_from = 0, descriptors = 0, fail_descriptor = 0, live = 0;
    bool createDeviceLocalBuffer(size_t n, BufferInfo &b) { return createBuffer(n, 0, 0, b); }
    bool createStagingBuffer(size_t n, BufferInfo &b) { return createBuffer(n, 0, 0, b); }
    bool createBuffer(size_t n, unsigned use, unsigned flags, BufferInfo &b) {
        (void)n; (void)use; (void)flags;
        if (++calls >= fail_from && fail_from) return false;
        b.buffer = (uintptr_t)++live; return true;
    }
    bool updateDescriptorSet(int pipeline, std::initializer_list<BufferInfo> buffers) {
        (void)pipeline;
        for (auto b : buffers) assert(b.buffer);
        return ++descriptors != fail_descriptor;
    }
    void destroyBuffer(BufferInfo b) { assert(b.buffer && live > 0); --live; }
};
struct vh_vk {
    void *d = nullptr;
    VulkanComputeRunner runner;
    VulkanComputeRunner::BufferInfo w{}, skin{}, out{}, stage{}, c_scr{}, rest{1}, morph{2}, sj{3}, sw{4}, c_ints{5}, c_flts{6};
    size_t cap = 0, M = 1, J = 1, V = 3;
    bool has_ct = true, bar_inputs = true, cached_readback = true;
    int ct_pc[26] = {}, pipe_ct = 1, pipe = 2;
    std::vector<float> hw, hskin, hscratch;
};
static size_t vh_deformer_prepare_scratch(void *, size_t frames) { return 4 * frames; }

#include "vhuman_ensure_under_test.inc"

static void cleanup(vh_vk &g) {
    for (auto *b : {&g.w, &g.skin, &g.out, &g.stage, &g.c_scr})
        if (b->buffer) { g.runner.destroyBuffer(*b); *b = {}; }
    assert(!g.runner.live);
}
int main() {
    // Both input and readback fallbacks also fail after the injected OOM.
    for (int nth = 1; nth <= 5; ++nth) {
        vh_vk g;
        assert(ensure(&g, 1));
        g.runner.fail_from = g.runner.calls + nth;
        assert(!ensure(&g, 65) && !g.cap);
        g.runner.fail_from = 0;
        assert(ensure(&g, 1) && g.cap == 64 && g.out.buffer && g.c_scr.buffer);
        cleanup(g);
    }
    for (int nth = 1; nth <= 2; ++nth) {
        vh_vk g;
        assert(ensure(&g, 1));
        g.runner.fail_descriptor = g.runner.descriptors + nth;
        assert(!ensure(&g, 65) && !g.cap);
        g.runner.fail_descriptor = 0;
        assert(ensure(&g, 1) && g.cap == 64);
        cleanup(g);
    }
    puts("Vulkan buffer retry fault injection passed (CPU doubles)");
}
