// SPDX-License-Identifier: MIT
// Standalone Vulkan 1.3 GEMM and device-local capacity probe. No Vulkan import library.
#include "vkew.h"
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GIB (UINT64_C(1) << 30)
#define STAGING_BYTES (UINT64_C(8) << 20)
#define RESERVE_BYTES (UINT64_C(512) << 20)
#define WINDOW_BYTES (UINT64_C(16) << 20)
#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif

typedef struct Buffer {
    VkBuffer handle;
    VkDeviceMemory memory;
    VkDeviceSize size;
    uint32_t heap;
    VkMemoryPropertyFlags flags;
    void *mapped;
} Buffer;

typedef struct Context {
    VkInstance instance;
    VkPhysicalDevice physical;
    VkDevice device;
    VkQueue queue;
    VkCommandPool pool;
    VkCommandBuffer cmd;
    VkFence fence;
    VkQueryPool queries;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties memory;
    VkPhysicalDeviceVulkan12Features f12;
    VkPhysicalDevice16BitStorageFeatures f16;
    VkPhysicalDeviceFeatures features;
    VkDeviceSize max_allocation, max_buffer;
    uint32_t family, timestamp_bits, local_heap;
    int has_budget, no_overallocation, has_shader_info;
    Buffer staging;
} Context;

typedef struct Pipeline {
    VkDescriptorSetLayout descriptor_layout;
    VkPipelineLayout layout;
    VkPipeline handle;
    VkDescriptorPool pool;
    VkDescriptorSet set;
    uint32_t tile;
    uint32_t tile_m;
    uint32_t pack_a;
} Pipeline;

typedef struct Options {
    uint32_t device, m, n, k, warmup, iterations;
    int type, info, vram;
    VkDeviceSize vram_bytes;
    uint32_t vram_reserve_mib;
    uint32_t fp32_register, fp32_tile, fp32_kstep, fp32_pad;
    uint32_t fp32_prefetch;
    uint32_t fp32_rows;
    uint32_t fp32_lds_prefetch;
    uint32_t fp32_pack_a;
    const char *fp32_isa;
} Options;

static const char *type_names[] = { "int8", "int16", "int32", "fp16", "fp32", "fp64" };
static const uint32_t input_bytes[] = { 1, 2, 4, 2, 4, 8 };
static const uint32_t output_bytes[] = { 4, 4, 4, 4, 4, 8 };

static int vk_ok(VkResult result, const char *operation) {
    if (result == VK_SUCCESS) return 1;
    fprintf(stderr, "%s: %s (%d)\n", operation, vkewResultToString(result), (int)result);
    return 0;
}
#define CHECK(call) do { if (!vk_ok((call), #call)) goto cleanup; } while (0)

static void *allocate_host(size_t bytes) {
    void *p = calloc(1, bytes);
    if (!p) fprintf(stderr, "Host allocation failed: %zu bytes\n", bytes);
    return p;
}

static VkDeviceSize min_size(VkDeviceSize a, VkDeviceSize b) { return a < b ? a : b; }

static void destroy_buffer(Context *ctx, Buffer *buffer) {
    if (buffer->mapped) vkUnmapMemory(ctx->device, buffer->memory);
    if (buffer->handle) vkDestroyBuffer(ctx->device, buffer->handle, NULL);
    if (buffer->memory) vkFreeMemory(ctx->device, buffer->memory, NULL);
    memset(buffer, 0, sizeof(*buffer));
}

// host=0 selects only the large device-local heap, never a system/aperture heap.
static int create_buffer(Context *ctx, Buffer *buffer, VkDeviceSize bytes,
                         VkBufferUsageFlags usage, int host) {
    VkBufferCreateInfo bi = {0};
    VkMemoryRequirements requirements;
    VkMemoryAllocateInfo ai = {0};
    uint32_t type = UINT32_MAX;
    int success = 0;
    if (!bytes || bytes > ctx->max_buffer ||
        ((usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) && bytes > ctx->props.limits.maxStorageBufferRange)) {
        fprintf(stderr, "Buffer size exceeds device limits: %" PRIu64 " bytes\n", bytes);
        return 0;
    }
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK(vkCreateBuffer(ctx->device, &bi, NULL, &buffer->handle));
    vkGetBufferMemoryRequirements(ctx->device, buffer->handle, &requirements);
    if (requirements.size > ctx->max_allocation) {
        fprintf(stderr, "Allocation requirements exceed maxMemoryAllocationSize\n");
        goto cleanup;
    }
    for (uint32_t i = 0; i < ctx->memory.memoryTypeCount; ++i) {
        VkMemoryType mt = ctx->memory.memoryTypes[i];
        if (!(requirements.memoryTypeBits & (1u << i))) continue;
        if (host) {
            if (!(mt.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
            if (type == UINT32_MAX) type = i;
            // Prefer coherent system memory, preserving the device-local budget.
            if (!(mt.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                (mt.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { type = i; break; }
        } else if (mt.heapIndex == ctx->local_heap &&
                   (mt.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { type = i; break; }
    }
    if (type == UINT32_MAX) { fprintf(stderr, "No suitable memory type\n"); goto cleanup; }
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = requirements.size;
    ai.memoryTypeIndex = type;
    CHECK(vkAllocateMemory(ctx->device, &ai, NULL, &buffer->memory));
    CHECK(vkBindBufferMemory(ctx->device, buffer->handle, buffer->memory, 0));
    buffer->size = bytes;
    buffer->heap = ctx->memory.memoryTypes[type].heapIndex;
    buffer->flags = ctx->memory.memoryTypes[type].propertyFlags;
    if (host) CHECK(vkMapMemory(ctx->device, buffer->memory, 0, VK_WHOLE_SIZE, 0, &buffer->mapped));
    success = 1;
cleanup:
    if (!success) destroy_buffer(ctx, buffer);
    return success;
}

static int sync_mapping(Context *ctx, Buffer *buffer, int invalidate) {
    VkMappedMemoryRange range = {0};
    if (buffer->flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) return 1;
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = buffer->memory;
    range.size = VK_WHOLE_SIZE;
    return vk_ok(invalidate ? vkInvalidateMappedMemoryRanges(ctx->device, 1, &range) :
                             vkFlushMappedMemoryRanges(ctx->device, 1, &range), "sync mapped memory");
}

static void barrier(Context *ctx) {
    VkMemoryBarrier memory = {0};
    memory.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memory.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    memory.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(ctx->cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &memory, 0, NULL, 0, NULL);
}

static int begin_commands(Context *ctx) {
    VkCommandBufferBeginInfo bi = {0};
    if (!vk_ok(vkResetCommandPool(ctx->device, ctx->pool, 0), "reset command pool")) return 0;
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (!vk_ok(vkBeginCommandBuffer(ctx->cmd, &bi), "begin command buffer")) return 0;
    barrier(ctx);
    return 1;
}

static int submit_commands(Context *ctx) {
    VkSubmitInfo submit = {0};
    if (!vk_ok(vkEndCommandBuffer(ctx->cmd), "end command buffer")) return 0;
    if (!vk_ok(vkResetFences(ctx->device, 1, &ctx->fence), "reset fence")) return 0;
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &ctx->cmd;
    if (!vk_ok(vkQueueSubmit(ctx->queue, 1, &submit, ctx->fence), "submit")) return 0;
    VkResult result = vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_C(60000000000));
    if (!vk_ok(result, "wait for GPU (60 second timeout)")) {
        // Avoid destroying resources still in flight following a timeout/device loss.
        fprintf(stderr, "Stopping process; GPU work did not complete.\n");
        exit(EXIT_FAILURE);
    }
    return 1;
}

static int transfer(Context *ctx, Buffer *gpu, VkDeviceSize offset,
                    void *host, VkDeviceSize bytes, int download) {
    unsigned char *data = (unsigned char *)host;
    if (offset > gpu->size || bytes > gpu->size - offset || (offset | bytes) % 4 != 0) {
        fprintf(stderr, "Invalid transfer range/alignment\n"); return 0;
    }
    for (VkDeviceSize done = 0; done < bytes;) {
        VkDeviceSize size = min_size(bytes - done, ctx->staging.size);
        VkBufferCopy copy = {0};
        if (!download) {
            memcpy(ctx->staging.mapped, data + (size_t)done, (size_t)size);
            if (!sync_mapping(ctx, &ctx->staging, 0)) return 0;
        }
        if (!begin_commands(ctx)) return 0;
        copy.size = size;
        if (download) {
            copy.srcOffset = offset + done;
            vkCmdCopyBuffer(ctx->cmd, gpu->handle, ctx->staging.handle, 1, &copy);
        } else {
            copy.dstOffset = offset + done;
            vkCmdCopyBuffer(ctx->cmd, ctx->staging.handle, gpu->handle, 1, &copy);
        }
        barrier(ctx);
        if (!submit_commands(ctx)) return 0;
        if (download) {
            if (!sync_mapping(ctx, &ctx->staging, 1)) return 0;
            memcpy(data + (size_t)done, ctx->staging.mapped, (size_t)size);
        }
        done += size;
    }
    return 1;
}

static void destroy_pipeline(Context *ctx, Pipeline *pipeline) {
    if (pipeline->handle) vkDestroyPipeline(ctx->device, pipeline->handle, NULL);
    if (pipeline->pool) vkDestroyDescriptorPool(ctx->device, pipeline->pool, NULL);
    if (pipeline->layout) vkDestroyPipelineLayout(ctx->device, pipeline->layout, NULL);
    if (pipeline->descriptor_layout) vkDestroyDescriptorSetLayout(ctx->device, pipeline->descriptor_layout, NULL);
    memset(pipeline, 0, sizeof(*pipeline));
}

static int create_pipeline(Context *ctx, Pipeline *pipeline, const char *file,
                           uint32_t bindings, uint32_t push_bytes, const VkSpecializationInfo *specialization) {
    VkDescriptorSetLayoutBinding binding[3] = {0};
    VkDescriptorSetLayoutCreateInfo di = {0};
    VkPushConstantRange push = {VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
    VkPipelineLayoutCreateInfo li = {0};
    VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bindings};
    VkDescriptorPoolCreateInfo pi = {0};
    VkDescriptorSetAllocateInfo ai = {0};
    VkShaderModuleCreateInfo si = {0};
    VkComputePipelineCreateInfo ci = {0};
    VkShaderModule shader = 0;
    char path[1024];
    FILE *stream = NULL;
    uint32_t *code = NULL;
    long length;
    int success = 0;
    if (snprintf(path, sizeof(path), "%s/%s", SHADER_DIR, file) >= (int)sizeof(path)) goto cleanup;
    stream = fopen(path, "rb");
    if (!stream) { fprintf(stderr, "Cannot open shader: %s\n", path); goto cleanup; }
    if (fseek(stream, 0, SEEK_END) != 0 || (length = ftell(stream)) <= 0 || length % 4 != 0 ||
        fseek(stream, 0, SEEK_SET) != 0) { fprintf(stderr, "Invalid SPIR-V file\n"); goto cleanup; }
    code = (uint32_t *)allocate_host((size_t)length);
    if (!code || fread(code, 1, (size_t)length, stream) != (size_t)length) goto cleanup;
    for (uint32_t i = 0; i < bindings; ++i) {
        binding[i].binding = i;
        binding[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding[i].descriptorCount = 1;
        binding[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    di.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    di.bindingCount = bindings;
    di.pBindings = binding;
    CHECK(vkCreateDescriptorSetLayout(ctx->device, &di, NULL, &pipeline->descriptor_layout));
    li.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    li.setLayoutCount = 1;
    li.pSetLayouts = &pipeline->descriptor_layout;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &push;
    CHECK(vkCreatePipelineLayout(ctx->device, &li, NULL, &pipeline->layout));
    pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.maxSets = 1;
    pi.poolSizeCount = 1;
    pi.pPoolSizes = &pool_size;
    CHECK(vkCreateDescriptorPool(ctx->device, &pi, NULL, &pipeline->pool));
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = pipeline->pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &pipeline->descriptor_layout;
    CHECK(vkAllocateDescriptorSets(ctx->device, &ai, &pipeline->set));
    si.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    si.codeSize = (size_t)length;
    si.pCode = code;
    CHECK(vkCreateShaderModule(ctx->device, &si, NULL, &shader));
    ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = shader;
    ci.stage.pName = "main";
    ci.stage.pSpecializationInfo = specialization;
    ci.layout = pipeline->layout;
    pipeline->tile = 16;
    pipeline->tile_m = 16;
    CHECK(vkCreateComputePipelines(ctx->device, 0, 1, &ci, NULL, &pipeline->handle));
    success = 1;
cleanup:
    if (shader) vkDestroyShaderModule(ctx->device, shader, NULL);
    free(code);
    if (stream) fclose(stream);
    if (!success) destroy_pipeline(ctx, pipeline);
    return success;
}

static void bind_buffers(Context *ctx, Pipeline *pipeline, Buffer **buffers, uint32_t count) {
    VkDescriptorBufferInfo info[3] = {0};
    VkWriteDescriptorSet writes[3] = {0};
    for (uint32_t i = 0; i < count; ++i) {
        info[i].buffer = buffers[i]->handle;
        info[i].range = buffers[i]->size;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = pipeline->set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &info[i];
    }
    vkUpdateDescriptorSets(ctx->device, count, writes, 0, NULL);
}

static void dispatch(Context *ctx, Pipeline *pipeline, const void *params, uint32_t bytes,
                     uint32_t x, uint32_t y) {
    vkCmdBindPipeline(ctx->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->handle);
    vkCmdBindDescriptorSets(ctx->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout,
                            0, 1, &pipeline->set, 0, NULL);
    vkCmdPushConstants(ctx->cmd, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, bytes, params);
    vkCmdDispatch(ctx->cmd, x, y, 1);
}

static int has_extension(VkExtensionProperties *extensions, uint32_t count, const char *name) {
    for (uint32_t i = 0; i < count; ++i)
        if (!strcmp(extensions[i].extensionName, name)) return 1;
    return 0;
}

static int query_budget(Context *ctx, VkPhysicalDeviceMemoryBudgetPropertiesEXT *budget) {
    VkPhysicalDeviceMemoryProperties2 properties = {0};
    memset(budget, 0, sizeof(*budget));
    if (!ctx->has_budget) return 0;
    budget->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    properties.pNext = budget;
    vkGetPhysicalDeviceMemoryProperties2(ctx->physical, &properties);
    return 1;
}

static int init_context(Context *ctx, uint32_t device_index, int info_only) {
    VkApplicationInfo app = {0};
    VkInstanceCreateInfo instance = {0};
    VkPhysicalDevice *devices = NULL;
    VkExtensionProperties *extensions = NULL;
    VkQueueFamilyProperties *queues = NULL;
    VkPhysicalDeviceFeatures2 features = {0};
    VkPhysicalDeviceProperties2 properties = {0};
    VkPhysicalDeviceMaintenance3Properties m3 = {0};
    VkPhysicalDeviceMaintenance4Properties m4 = {0};
    VkPhysicalDeviceVulkan12Features enabled12 = {0};
    VkPhysicalDevice16BitStorageFeatures enabled16 = {0};
    VkPhysicalDeviceFeatures enabled = {0};
    VkDeviceMemoryOverallocationCreateInfoAMD overallocation = {0};
    VkDeviceQueueCreateInfo queue = {0};
    VkDeviceCreateInfo device = {0};
    VkCommandPoolCreateInfo pool = {0};
    VkCommandBufferAllocateInfo command = {0};
    VkFenceCreateInfo fence = {0};
    VkQueryPoolCreateInfo queries = {0};
    const char *enabled_extensions[3];
    uint32_t count = 0, extension_count = 0, queue_count = 0, enabled_count = 0;
    float priority = 1.0f;
    int success = 0;
    if (!vkewInit()) { fprintf(stderr, "%s\n", vkewGetError()); goto cleanup; }
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "vulkan-gemm";
    app.apiVersion = VK_API_VERSION_1_3;
    instance.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance.pApplicationInfo = &app;
    CHECK(vkCreateInstance(&instance, NULL, &ctx->instance));
    if (!vkewLoadInstance(ctx->instance)) { fprintf(stderr, "%s\n", vkewGetError()); goto cleanup; }
    CHECK(vkEnumeratePhysicalDevices(ctx->instance, &count, NULL));
    if (device_index >= count) { fprintf(stderr, "Invalid device index %u (count %u)\n", device_index, count); goto cleanup; }
    devices = (VkPhysicalDevice *)allocate_host(sizeof(*devices) * count);
    if (!devices) goto cleanup;
    CHECK(vkEnumeratePhysicalDevices(ctx->instance, &count, devices));
    for (uint32_t i = 0; i < count; ++i) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devices[i], &p);
        printf("device[%u]: %s%s\n", i, p.deviceName, i == device_index ? " (selected)" : "");
    }
    ctx->physical = devices[device_index];
    vkGetPhysicalDeviceProperties(ctx->physical, &ctx->props);
    if (ctx->props.apiVersion < VK_API_VERSION_1_3 || !vkGetPhysicalDeviceFeatures2 ||
        !vkGetPhysicalDeviceProperties2 || !vkGetPhysicalDeviceMemoryProperties2) {
        fprintf(stderr, "Vulkan 1.3 required\n"); goto cleanup;
    }
    printf("API %u.%u.%u vendor=0x%04x device=0x%04x driver=0x%08x\n",
           VK_VERSION_MAJOR(ctx->props.apiVersion), VK_VERSION_MINOR(ctx->props.apiVersion),
           VK_VERSION_PATCH(ctx->props.apiVersion), ctx->props.vendorID, ctx->props.deviceID, ctx->props.driverVersion);
    ctx->f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    ctx->f16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    ctx->f12.pNext = &ctx->f16;
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &ctx->f12;
    vkGetPhysicalDeviceFeatures2(ctx->physical, &features);
    ctx->features = features.features;
    m3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
    m4.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES;
    m3.pNext = &m4;
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &m3;
    vkGetPhysicalDeviceProperties2(ctx->physical, &properties);
    ctx->max_allocation = m3.maxMemoryAllocationSize;
    ctx->max_buffer = m4.maxBufferSize;
    CHECK(vkEnumerateDeviceExtensionProperties(ctx->physical, NULL, &extension_count, NULL));
    extensions = (VkExtensionProperties *)allocate_host(sizeof(*extensions) * extension_count);
    if (!extensions) goto cleanup;
    CHECK(vkEnumerateDeviceExtensionProperties(ctx->physical, NULL, &extension_count, extensions));
    ctx->has_budget = has_extension(extensions, extension_count, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    ctx->has_shader_info = has_extension(extensions, extension_count, VK_AMD_SHADER_INFO_EXTENSION_NAME);
    ctx->no_overallocation = has_extension(extensions, extension_count,
                                          VK_AMD_MEMORY_OVERALLOCATION_BEHAVIOR_EXTENSION_NAME);
    vkGetPhysicalDeviceMemoryProperties(ctx->physical, &ctx->memory);
    ctx->local_heap = UINT32_MAX;
    for (uint32_t i = 0; i < ctx->memory.memoryHeapCount; ++i) {
        if ((ctx->memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
            (ctx->local_heap == UINT32_MAX || ctx->memory.memoryHeaps[i].size >
             ctx->memory.memoryHeaps[ctx->local_heap].size)) ctx->local_heap = i;
    }
    if (ctx->local_heap == UINT32_MAX) { fprintf(stderr, "No device-local heap\n"); goto cleanup; }
    printf("features: int8=%u storage8=%u int16=%u storage16=%u fp16_arithmetic=%u fp64=%u\n",
           ctx->f12.shaderInt8, ctx->f12.storageBuffer8BitAccess, ctx->features.shaderInt16,
           ctx->f16.storageBuffer16BitAccess, ctx->f12.shaderFloat16, ctx->features.shaderFloat64);
    printf("int32/fp32: core; fp16 path: storage -> fp32 multiply/accumulate/output\n");
    if (!ctx->f12.shaderFloat16) printf("native fp16 arithmetic: UNSUPPORTED\n");
    printf("limits: allocation=%" PRIu64 " buffer=%" PRIu64 " storage_range=%u shared=%u bytes\n",
           ctx->max_allocation, ctx->max_buffer, ctx->props.limits.maxStorageBufferRange,
           ctx->props.limits.maxComputeSharedMemorySize);
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget;
    int budget_available = query_budget(ctx, &budget);
    for (uint32_t i = 0; i < ctx->memory.memoryHeapCount; ++i) {
        printf("heap[%u]: size=%.3f GiB flags=0x%x%s", i,
               (double)ctx->memory.memoryHeaps[i].size / GIB, ctx->memory.memoryHeaps[i].flags,
               i == ctx->local_heap ? " (selected VRAM)" : "");
        if (budget_available) printf(" budget=%.3f usage=%.3f GiB", (double)budget.heapBudget[i]/GIB,
                                     (double)budget.heapUsage[i]/GIB);
        printf("\n");
    }
    if (info_only) { success = 1; goto cleanup; }
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical, &queue_count, NULL);
    queues = (VkQueueFamilyProperties *)allocate_host(sizeof(*queues) * queue_count);
    if (!queues) goto cleanup;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical, &queue_count, queues);
    ctx->family = UINT32_MAX;
    for (uint32_t i = 0; i < queue_count; ++i) {
        if (queues[i].queueCount && (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && queues[i].timestampValidBits) {
            ctx->family = i;
            if (!(queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) break;
        }
    }
    if (ctx->family == UINT32_MAX) { fprintf(stderr, "No compute queue with timestamps\n"); goto cleanup; }
    ctx->timestamp_bits = queues[ctx->family].timestampValidBits;
    if (ctx->props.limits.maxComputeWorkGroupInvocations < 256 ||
        ctx->props.limits.maxComputeWorkGroupSize[0] < 256 ||
        ctx->props.limits.maxComputeWorkGroupSize[1] < 16 || ctx->props.limits.maxComputeSharedMemorySize < 4096) {
        fprintf(stderr, "Insufficient workgroup/shared-memory limits\n"); goto cleanup;
    }
    enabled.shaderInt16 = ctx->features.shaderInt16;
    enabled.shaderFloat64 = ctx->features.shaderFloat64;
    enabled12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    enabled12.storageBuffer8BitAccess = ctx->f12.storageBuffer8BitAccess;
    enabled12.shaderInt8 = ctx->f12.shaderInt8;
    enabled16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    enabled16.storageBuffer16BitAccess = ctx->f16.storageBuffer16BitAccess;
    enabled12.pNext = &enabled16;
    if (ctx->has_budget) enabled_extensions[enabled_count++] = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
    if (ctx->has_shader_info) enabled_extensions[enabled_count++] = VK_AMD_SHADER_INFO_EXTENSION_NAME;
    if (ctx->no_overallocation) {
        enabled_extensions[enabled_count++] = VK_AMD_MEMORY_OVERALLOCATION_BEHAVIOR_EXTENSION_NAME;
        overallocation.sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_OVERALLOCATION_CREATE_INFO_AMD;
        overallocation.overallocationBehavior = VK_MEMORY_OVERALLOCATION_BEHAVIOR_DISALLOWED_AMD;
        enabled16.pNext = &overallocation;
    }
    queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue.queueFamilyIndex = ctx->family;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    device.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device.pNext = &enabled12;
    device.pEnabledFeatures = &enabled;
    device.queueCreateInfoCount = 1;
    device.pQueueCreateInfos = &queue;
    device.enabledExtensionCount = enabled_count;
    device.ppEnabledExtensionNames = enabled_extensions;
    CHECK(vkCreateDevice(ctx->physical, &device, NULL, &ctx->device));
    if (!vkewLoadDevice(ctx->device)) { fprintf(stderr, "%s\n", vkewGetError()); goto cleanup; }
    vkGetDeviceQueue(ctx->device, ctx->family, 0, &ctx->queue);
    pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.queueFamilyIndex = ctx->family;
    CHECK(vkCreateCommandPool(ctx->device, &pool, NULL, &ctx->pool));
    command.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command.commandPool = ctx->pool;
    command.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command.commandBufferCount = 1;
    CHECK(vkAllocateCommandBuffers(ctx->device, &command, &ctx->cmd));
    fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    CHECK(vkCreateFence(ctx->device, &fence, NULL, &ctx->fence));
    queries.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queries.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queries.queryCount = 2;
    CHECK(vkCreateQueryPool(ctx->device, &queries, NULL, &ctx->queries));
    if (!create_buffer(ctx, &ctx->staging, STAGING_BYTES,
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 1)) goto cleanup;
    printf("compute queue=%u timestamp_bits=%u period=%.3f ns; overallocation=%s\n",
           ctx->family, ctx->timestamp_bits, ctx->props.limits.timestampPeriod,
           ctx->no_overallocation ? "disallowed" : "driver default");
    success = 1;
cleanup:
    free(devices); free(extensions); free(queues);
    return success;
}

static void destroy_context(Context *ctx) {
    if (ctx->device) {
        if (vkDeviceWaitIdle) vkDeviceWaitIdle(ctx->device);
        if (ctx->staging.handle) destroy_buffer(ctx, &ctx->staging);
        if (ctx->queries) vkDestroyQueryPool(ctx->device, ctx->queries, NULL);
        if (ctx->fence) vkDestroyFence(ctx->device, ctx->fence, NULL);
        if (ctx->pool) vkDestroyCommandPool(ctx->device, ctx->pool, NULL);
        if (vkDestroyDevice) vkDestroyDevice(ctx->device, NULL);
    }
    if (ctx->instance && vkDestroyInstance) vkDestroyInstance(ctx->instance, NULL);
    vkewShutdown();
}

static int supported(Context *ctx, int type) {
    switch (type) {
        case 0: return ctx->f12.shaderInt8 && ctx->f12.storageBuffer8BitAccess;
        case 1: return ctx->features.shaderInt16 && ctx->f16.storageBuffer16BitAccess;
        case 3: return ctx->f16.storageBuffer16BitAccess;
        case 5: return ctx->features.shaderFloat64;
        default: return 1;
    }
}

// Deterministic, bounded inputs. K <= INT32_MAX/256 makes all integer sums safe.
static int input_integer(uint32_t index, uint32_t seed) {
    uint32_t v = index ^ seed;
    v = (v ^ (v >> 16)) * 0x7feb352du;
    v = (v ^ (v >> 15)) * 0x846ca68bu;
    return (int)((v ^ (v >> 16)) % 33u) - 16;
}

// Generated nonzero FP16 inputs are normal, finite and in [-1,1].
static uint16_t input_half(float value) {
    uint32_t bits, mantissa, rounded;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffu) == 0) return 0;
    mantissa = bits & 0x7fffffu;
    rounded = (mantissa + 0xfffu + ((mantissa >> 13) & 1u)) >> 13;
    return (uint16_t)(((bits >> 16) & 0x8000u) + ((((bits >> 23) & 255u) - 112u) << 10) + rounded);
}

static double input_value(int type, uint32_t index, uint32_t seed) {
    int integer = input_integer(index, seed);
    if (type < 3) return integer;
    if (type == 5) return (double)integer / 17.0;
    float value = (float)integer / 17.0f;
    if (type == 3) {
        uint16_t half = input_half(value);
        if ((half & 0x7fffu) == 0) return 0;
        double result = ldexp(1.0 + (half & 1023u) / 1024.0, (int)((half >> 10) & 31u) - 15);
        return half & 0x8000u ? -result : result;
    }
    return value;
}

static void fill_inputs(void *data, int type, uint32_t count, uint32_t seed) {
    for (uint32_t i = 0; i < count; ++i) {
        double v = input_value(type, i, seed);
        switch (type) {
            case 0: ((int8_t *)data)[i] = (int8_t)v; break;
            case 1: ((int16_t *)data)[i] = (int16_t)v; break;
            case 2: ((int32_t *)data)[i] = (int32_t)v; break;
            case 3: ((uint16_t *)data)[i] = input_half((float)v); break;
            case 4: ((float *)data)[i] = (float)v; break;
            case 5: ((double *)data)[i] = v; break;
        }
    }
}

static int check_output(Context *ctx, Buffer *output, int type, uint32_t m, uint32_t n, uint32_t k) {
    uint64_t elements = (uint64_t)m * n;
    int full_check = elements <= 4096 || (elements <= 65536 && k <= 128);
    uint32_t samples = full_check ? (uint32_t)elements : 64u;
    unsigned char *all = NULL;
    double max_error = 0;
    int success = 0;
    if (full_check) {
        all = (unsigned char *)allocate_host((size_t)elements * output_bytes[type]);
        if (!all || !transfer(ctx, output, 0, all, elements * output_bytes[type], 1)) goto cleanup;
    }
    for (uint32_t sample = 0; sample < samples; ++sample) {
        uint64_t index = all ? sample : ((uint64_t)sample * (elements - 1)) / (samples - 1);
        uint32_t row = (uint32_t)(index / n), col = (uint32_t)(index % n);
        double reference = 0, magnitude = 0, actual;
        union { int32_t integer; float fp32; double fp64; } value;
        for (uint32_t i = 0; i < k; ++i) {
            double product = input_value(type, row*k+i, 123u) * input_value(type, i*n+col, 456u);
            reference += product;
            magnitude += fabs(product);
        }
        if (all) memcpy(&value, all + (size_t)index * output_bytes[type], output_bytes[type]);
        else if (!transfer(ctx, output, index * output_bytes[type], &value, output_bytes[type], 1)) goto cleanup;
        actual = type < 3 ? value.integer : (type == 5 ? value.fp64 : value.fp32);
        double tolerance = type < 3 ? 0 : (type == 5 ? 1e-12 : 2e-6) * fmax(1.0, magnitude);
        double error = fabs(actual - reference);
        if (!isfinite(actual) || error > tolerance) {
            fprintf(stderr, "%s mismatch [%u,%u]: GPU=%.17g CPU=%.17g abs_error=%.3g tolerance=%.3g\n",
                    type_names[type], row, col, actual, reference, error, tolerance);
            goto cleanup;
        }
        if (error > max_error) max_error = error;
    }
    printf("  correctness PASS (%u %s outputs, max_abs_error=%.3g)\n", samples,
           all ? "all" : "sampled", max_error);
    success = 1;
cleanup:
    free(all);
    return success;
}

static int pack_fp32(Context *ctx, Buffer *input, Buffer *packed, uint32_t m, uint32_t k, double *ms) {
    Pipeline pipeline = {0};
    Buffer *buffers[] = {input, packed};
    uint32_t params[] = {m, k};
    uint32_t x = (k-1)/32+1, y = (m-1)/32+1;
    uint64_t timestamps[2];
    int success = 0;
    if (x > ctx->props.limits.maxComputeWorkGroupCount[0] || y > ctx->props.limits.maxComputeWorkGroupCount[1]) {
        fprintf(stderr, "Packing dispatch exceeds device limits\n"); return 0;
    }
    if (!create_pipeline(ctx, &pipeline, "pack_fp32.spv", 2, 8, NULL)) return 0;
    bind_buffers(ctx, &pipeline, buffers, 2);
    if (!begin_commands(ctx)) goto cleanup;
    vkCmdResetQueryPool(ctx->cmd, ctx->queries, 0, 2);
    vkCmdWriteTimestamp(ctx->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx->queries, 0);
    dispatch(ctx, &pipeline, params, sizeof(params), x, y);
    vkCmdWriteTimestamp(ctx->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->queries, 1);
    barrier(ctx);
    if (!submit_commands(ctx)) goto cleanup;
    CHECK(vkGetQueryPoolResults(ctx->device, ctx->queries, 0, 2, sizeof(timestamps), timestamps,
                               sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    uint64_t delta = timestamps[1]-timestamps[0];
    if (ctx->timestamp_bits < 64) delta &= (UINT64_C(1) << ctx->timestamp_bits)-1;
    *ms = (double)delta*ctx->props.limits.timestampPeriod/1e6;
    success = 1;
cleanup:
    destroy_pipeline(ctx, &pipeline);
    return success;
}

static int gemm(Context *ctx, Pipeline *pipeline, int type, uint32_t m, uint32_t n, uint32_t k,
                uint32_t warmup, uint32_t iterations) {
    Buffer a = {0}, b = {0}, c = {0}, packed_a = {0};
    Buffer *buffers[] = {&a, &b, &c};
    uint32_t params[] = {m, n, k};
    uint64_t na = (uint64_t)m*k, nb = (uint64_t)k*n, nc = (uint64_t)m*n;
    uint64_t sizes[] = {(na*input_bytes[type]+3u)&~UINT64_C(3),
                        (nb*input_bytes[type]+3u)&~UINT64_C(3), nc*output_bytes[type]};
    void *host = NULL;
    double sum_ms = 0, min_ms = HUGE_VAL, max_ms = 0, packing_ms = 0;
    int success = 0;
    uint32_t groups_x = (n-1u)/pipeline->tile + 1u;
    uint32_t groups_y = (m-1u)/pipeline->tile_m + 1u;
    if (na > UINT32_MAX || nb > UINT32_MAX || nc > UINT32_MAX ||
        (type < 3 && k > INT32_MAX / 256u) ||
        groups_x > ctx->props.limits.maxComputeWorkGroupCount[0] ||
        groups_y > ctx->props.limits.maxComputeWorkGroupCount[1]) {
        fprintf(stderr, "Matrix dimensions exceed indexing, dispatch, or integer accumulation limits\n"); return 0;
    }
    for (uint32_t i = 0; i < 3; ++i) {
        if (sizes[i] > ctx->max_buffer || sizes[i] > ctx->max_allocation ||
            sizes[i] > ctx->props.limits.maxStorageBufferRange || sizes[i] > SIZE_MAX) {
            fprintf(stderr, "Matrix buffer exceeds device/host limits\n"); return 0;
        }
    }
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget;
    if (query_budget(ctx, &budget)) {
        VkDeviceSize available = budget.heapBudget[ctx->local_heap] > budget.heapUsage[ctx->local_heap] ?
            budget.heapBudget[ctx->local_heap] - budget.heapUsage[ctx->local_heap] : 0;
        if (available < RESERVE_BYTES || sizes[0]+sizes[1]+sizes[2]+(pipeline->pack_a ? sizes[0] : 0) > available-RESERVE_BYTES) {
            fprintf(stderr, "Matrices exceed live VRAM budget minus 512 MiB reserve\n"); return 0;
        }
    }
    printf("%s M=%u N=%u K=%u input=%s arithmetic=%s output=%s\n", type_names[type], m, n, k,
           type_names[type], type < 3 ? "int32" : (type == 5 ? "fp64" : "fp32"),
           type < 3 ? "int32" : (type == 5 ? "fp64" : "fp32"));
    for (uint32_t i = 0; i < 3; ++i) {
        if (!create_buffer(ctx, buffers[i], sizes[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0)) goto cleanup;
        if (i == 2) continue;
        host = allocate_host((size_t)sizes[i]);
        if (!host) goto cleanup;
        fill_inputs(host, type, (uint32_t)(i == 0 ? na : nb), i == 0 ? 123u : 456u);
        if (!transfer(ctx, buffers[i], 0, host, sizes[i], 0)) goto cleanup;
        free(host); host = NULL;
    }
    if (pipeline->pack_a) {
        if (!create_buffer(ctx, &packed_a, sizes[0], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0) ||
            !pack_fp32(ctx, &a, &packed_a, m, k, &packing_ms)) goto cleanup;
        buffers[0] = &packed_a;
        printf("  GPU A packing=%.4f ms (once, excluded from GEMM-only timing)\n", packing_ms);
    }
    bind_buffers(ctx, pipeline, buffers, 3);
    for (uint32_t i = 0; i < warmup + iterations; ++i) {
        uint64_t timestamps[2];
        if (!begin_commands(ctx)) goto cleanup;
        vkCmdResetQueryPool(ctx->cmd, ctx->queries, 0, 2);
        vkCmdWriteTimestamp(ctx->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx->queries, 0);
        dispatch(ctx, pipeline, params, sizeof(params), groups_x, groups_y);
        vkCmdWriteTimestamp(ctx->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->queries, 1);
        barrier(ctx);
        if (!submit_commands(ctx)) goto cleanup;
        CHECK(vkGetQueryPoolResults(ctx->device, ctx->queries, 0, 2, sizeof(timestamps), timestamps,
                                   sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        uint64_t delta = timestamps[1] - timestamps[0];
        if (ctx->timestamp_bits < 64) delta &= (UINT64_C(1) << ctx->timestamp_bits) - 1;
        double ms = (double)delta * ctx->props.limits.timestampPeriod / 1e6;
        if (i >= warmup) {
            sum_ms += ms;
            if (ms < min_ms) min_ms = ms;
            if (ms > max_ms) max_ms = ms;
        }
    }
    if (!check_output(ctx, &c, type, m, n, k)) goto cleanup;
    if (!(sum_ms > 0)) { fprintf(stderr, "Zero GPU timestamp duration\n"); goto cleanup; }
    printf("  GPU mean=%.4f ms min=%.4f max=%.4f; %.3f %s (%u iterations, %u warmups)\n",
           sum_ms/iterations, min_ms, max_ms,
           2.0*m*n*k / ((sum_ms/iterations)*1e6), type < 3 ? "GOP/s" : "GFLOP/s", iterations, warmup);
    if (pipeline->pack_a) printf("  GPU pack + one GEMM=%.4f ms; %.3f GFLOP/s\n",
        packing_ms+sum_ms/iterations, 2.0*m*n*k / ((packing_ms+sum_ms/iterations)*1e6));
    success = 1;
cleanup:
    free(host);
    destroy_buffer(ctx, &packed_a); destroy_buffer(ctx, &c); destroy_buffer(ctx, &b); destroy_buffer(ctx, &a);
    return success;
}

static int shader_info(Context *ctx, Pipeline *pipeline, const char *isa_file) {
    VkShaderStatisticsInfoAMD statistics = {0};
    size_t bytes = sizeof(statistics);
    if (!ctx->has_shader_info || !vkGetShaderInfoAMD) {
        if (isa_file) fprintf(stderr, "VK_AMD_shader_info unavailable\n");
        return isa_file == NULL;
    }
    VkResult status = vkGetShaderInfoAMD(ctx->device, pipeline->handle, VK_SHADER_STAGE_COMPUTE_BIT,
                                        VK_SHADER_INFO_TYPE_STATISTICS_AMD, &bytes, &statistics);
    if (status == VK_SUCCESS) printf("  AMD shader: VGPR=%u SGPR=%u LDS=%zu scratch=%zu bytes\n",
        statistics.resourceUsage.numUsedVgprs, statistics.resourceUsage.numUsedSgprs,
        statistics.resourceUsage.ldsUsageSizeInBytes, statistics.resourceUsage.scratchMemUsageInBytes);
    if (!isa_file) return 1;
    bytes = 0;
    if (!vk_ok(vkGetShaderInfoAMD(ctx->device, pipeline->handle, VK_SHADER_STAGE_COMPUTE_BIT,
        VK_SHADER_INFO_TYPE_DISASSEMBLY_AMD, &bytes, NULL), "query AMD shader disassembly size")) return 0;
    char *text = (char *)allocate_host(bytes);
    if (!text) return 0;
    int ok = vk_ok(vkGetShaderInfoAMD(ctx->device, pipeline->handle, VK_SHADER_STAGE_COMPUTE_BIT,
        VK_SHADER_INFO_TYPE_DISASSEMBLY_AMD, &bytes, text), "query AMD shader disassembly");
    if (ok) {
        FILE *file = fopen(isa_file, "wb");
        if (!file) { fprintf(stderr, "Cannot write %s\n", isa_file); ok = 0; }
        else {
            if (bytes && text[bytes-1] == '\0') --bytes;
            ok = fwrite(text, 1, bytes, file) == bytes;
            if (fclose(file) != 0) ok = 0;
        }
    }
    free(text);
    return ok;
}

static int benchmark(Context *ctx, const Options *options) {
    for (int type = 0; type < 6; ++type) {
        Pipeline pipeline = {0};
        char shader[32];
        if (options->type >= 0 && options->type != type) continue;
        if (!supported(ctx, type)) {
            printf("%s: UNSUPPORTED (required arithmetic/storage feature missing)\n", type_names[type]);
            if (options->type == type) return 0;
            continue;
        }
        snprintf(shader, sizeof(shader), "gemm_%d.spv", type);
        uint32_t config[] = { options->fp32_rows/16, options->fp32_kstep, options->fp32_pad, 0,
                             options->fp32_prefetch, options->fp32_tile/16, options->fp32_lds_prefetch,
                             options->fp32_pack_a };
        VkSpecializationMapEntry entries[] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}, {3, 12, 4}, {4, 16, 4}, {5, 20, 4}, {6, 24, 4}, {7, 28, 4}};
        VkSpecializationInfo spec = {8, entries, sizeof(config), config};
        int register_kernel = type == 4 && options->fp32_register;
        if (register_kernel) {
            uint32_t shared_bytes = config[1]*(options->fp32_rows+options->fp32_tile+4*config[2])*4;
            if (shared_bytes > ctx->props.limits.maxComputeSharedMemorySize) {
                fprintf(stderr, "FP32 tile exceeds shared-memory limit\n"); return 0;
            }
        }
        if (!create_pipeline(ctx, &pipeline, register_kernel ? "gemm_fp32_register.spv" : shader,
                             3, 12, register_kernel ? &spec : NULL)) return 0;
        if (register_kernel) {
            pipeline.tile = options->fp32_tile; pipeline.tile_m = options->fp32_rows;
            pipeline.pack_a = options->fp32_pack_a;
        }
        if (type == 4) printf("FP32 kernel=%s tile=%ux%u Kstep=%u padding=%u prefetch=%u\n",
            register_kernel ? "register" : "baseline", pipeline.tile_m, pipeline.tile,
            register_kernel ? config[1] : 16, register_kernel ? config[2] : 0,
            register_kernel ? config[4] : 0);
        int ok = gemm(ctx, &pipeline, type, 19, 23, 29, 0, 1);
        if (ok && register_kernel && options->m % pipeline.tile_m == 0 &&
            options->n % pipeline.tile == 0 && options->k % config[1] == 0) {
            destroy_pipeline(ctx, &pipeline);
            config[3] = 1;
            ok = create_pipeline(ctx, &pipeline, "gemm_fp32_register.spv", 3, 12, &spec);
            pipeline.tile = options->fp32_tile;
            pipeline.tile_m = options->fp32_rows;
            pipeline.pack_a = options->fp32_pack_a;
            printf("  FP32 aligned vector-load/store specialization\n");
        }
        if (ok && type == 4) ok = shader_info(ctx, &pipeline, options->fp32_isa);
        if (ok) ok = gemm(ctx, &pipeline, type, options->m, options->n, options->k,
                          options->warmup, options->iterations);
        destroy_pipeline(ctx, &pipeline);
        if (!ok) return 0;
    }
    return 1;
}

static int vram_test(Context *ctx, VkDeviceSize requested, VkDeviceSize reserve) {
    Buffer *chunks = NULL, result = {0};
    Pipeline pipeline = {0};
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget;
    VkDeviceSize allocated = 0, target, chunk_bytes, verified = 0;
    uint32_t count = 0, capacity;
    int success = 0;
    if (!query_budget(ctx, &budget)) {
        fprintf(stderr, "VRAM test requires VK_EXT_memory_budget\n"); return 0;
    }
    VkDeviceSize available = budget.heapBudget[ctx->local_heap] > budget.heapUsage[ctx->local_heap] ?
        budget.heapBudget[ctx->local_heap] - budget.heapUsage[ctx->local_heap] : 0;
    target = min_size(requested, available > reserve ? available - reserve : 0);
    target &= ~UINT64_C(3);
    chunk_bytes = min_size(GIB, min_size(ctx->max_allocation, ctx->max_buffer));
    chunk_bytes = min_size(chunk_bytes, ctx->props.limits.maxStorageBufferRange) & ~UINT64_C(3);
    printf("VRAM requested=%.3f GiB budget=%.3f usage=%.3f reserve=%.3f target=%.3f chunk=%.3f GiB\n",
           (double)requested/GIB, (double)budget.heapBudget[ctx->local_heap]/GIB,
           (double)budget.heapUsage[ctx->local_heap]/GIB, (double)reserve/GIB,
           (double)target/GIB, (double)chunk_bytes/GIB);
    if (!target || !chunk_bytes) { fprintf(stderr, "Insufficient VRAM budget\n"); return 0; }
    uint64_t needed = (target + chunk_bytes - 1) / chunk_bytes;
    if (needed + 2 > ctx->props.limits.maxMemoryAllocationCount || needed > UINT32_MAX) {
        fprintf(stderr, "VRAM test exceeds allocation count limit\n"); return 0;
    }
    // Permit partial final chunks as the driver's budget changes during allocation.
    capacity = ctx->props.limits.maxMemoryAllocationCount - 2;
    chunks = (Buffer *)allocate_host(sizeof(*chunks) * capacity);
    if (!chunks || !create_pipeline(ctx, &pipeline, "vram.spv", 2, 20, NULL) ||
        !create_buffer(ctx, &result, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0)) goto cleanup;
    while (allocated < target && count < capacity) {
        query_budget(ctx, &budget);
        VkDeviceSize budget_bytes = budget.heapBudget[ctx->local_heap];
        VkDeviceSize usage = budget.heapUsage[ctx->local_heap];
        available = budget_bytes > usage ? budget_bytes - usage : 0;
        VkDeviceSize bytes = min_size(chunk_bytes, target - allocated);
        if (available <= reserve || available - reserve < (UINT64_C(1) << 20)) {
            fprintf(stderr, "Live budget shrank; stopping allocation at %.3f GiB\n", (double)allocated/GIB);
            break;
        }
        bytes = min_size(bytes, available - reserve) & ~UINT64_C(3);
        if (!create_buffer(ctx, &chunks[count], bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0)) break;
        allocated += bytes; ++count;
        printf("  allocated %u chunks: %.3f GiB\n", count, (double)allocated/GIB);
        fflush(stdout);
    }
    // Keep ALL allocations alive. Write all chunks before reading any, detecting aliases.
    for (uint32_t pattern = 0; pattern < 2; ++pattern) {
        for (uint32_t mode = 0; mode < 2; ++mode) {
            for (uint32_t chunk = 0; chunk < count; ++chunk) {
                Buffer *buffers[] = {&chunks[chunk], &result};
                uint32_t errors = 0;
                bind_buffers(ctx, &pipeline, buffers, 2);
                for (VkDeviceSize offset = 0; offset < chunks[chunk].size; offset += WINDOW_BYTES) {
                    VkDeviceSize bytes = min_size(WINDOW_BYTES, chunks[chunk].size - offset);
                    uint32_t params[] = {(uint32_t)(offset/4), (uint32_t)(bytes/4), chunk,
                                         pattern ? UINT32_MAX : 0, mode};
                    if (!begin_commands(ctx)) goto cleanup;
                    if (mode) {
                        vkCmdFillBuffer(ctx->cmd, result.handle, 0, 4, 0);
                        barrier(ctx);
                    }
                    uint32_t groups = ctx->props.limits.maxComputeWorkGroupCount[0];
                    if (groups > 4096) groups = 4096;
                    dispatch(ctx, &pipeline, params, sizeof(params), groups, 1);
                    barrier(ctx);
                    if (!submit_commands(ctx)) goto cleanup;
                    if (mode) {
                        if (!transfer(ctx, &result, 0, &errors, sizeof(errors), 1)) goto cleanup;
                        if (errors) {
                            fprintf(stderr, "VRAM mismatch chunk=%u offset=%" PRIu64 " errors=%u\n", chunk, offset, errors);
                            goto cleanup;
                        }
                    }
                }
                printf("  pattern=%u %s chunk=%u PASS\n", pattern, mode ? "verify" : "write", chunk);
                fflush(stdout);
            }
        }
        verified = allocated;
        printf("  pattern %u: %.3f GiB fully written and verified\n", pattern, (double)verified/GIB);
    }
    query_budget(ctx, &budget);
    printf("VRAM %s requested=%" PRIu64 " allocated=%" PRIu64 " verified=%" PRIu64
           " bytes; final budget=%.3f usage=%.3f GiB\n",
           verified == requested ? "PASS" : "LIMITED", requested, allocated, verified,
           (double)budget.heapBudget[ctx->local_heap]/GIB, (double)budget.heapUsage[ctx->local_heap]/GIB);
    printf("Verified Vulkan device-local allocations and GPU access; physical residency is not guaranteed by Vulkan.\n");
    success = verified == requested;
cleanup:
    destroy_pipeline(ctx, &pipeline);
    destroy_buffer(ctx, &result);
    for (uint32_t i = 0; i < count; ++i) destroy_buffer(ctx, &chunks[i]);
    free(chunks);
    return success;
}

static void usage(const char *program) {
    printf("Usage: %s [--info] [--device INDEX] [--type all|int8|int16|int32|fp16|fp32|fp64]\n"
           "  [--m M] [--n N] [--k K] [--warmup W] [--iterations I]\n"
           "  [--fp32-kernel baseline|register] [--fp32-tile 64|128]\n"
           "  [--fp32-kstep 8|16|32] [--fp32-pad 0|1]\n"
           "  [--fp32-prefetch 0|1]\n"
           "  [--fp32-rows 64|128] (default matches --fp32-tile)\n"
           "  [--fp32-lds-prefetch 0|1]\n"
           "  [--fp32-pack-a 0|1] (GPU packing cost reported separately)\n"
           "  [--fp32-isa FILE] (AMD shader disassembly)\n"
           "  [--vram-test [GiB]] [--vram-reserve-mib MiB]\n"
           "Defaults: device=0 all types M=N=K=1024 warmup=2 iterations=10; VRAM target=14 GiB.\n"
           "VRAM reserve defaults to 512 MiB; --vram-reserve-mib adjusts the live-budget headroom.\n"
           "FP16 uses half inputs with FP32 multiplication, accumulation, and output.\n"
           "Exit: 0=PASS, 1=runtime/correctness/capacity failure, 2=invalid arguments.\n", program);
}

static int parse_uint(const char *text, uint32_t *out, int allow_zero) {
    char *end;
    if (!text[0] || text[0] < '0' || text[0] > '9') return 0;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || *end || value > UINT32_MAX - 15u || (!allow_zero && !value)) return 0;
    *out = (uint32_t)value;
    return 1;
}

int main(int argc, char **argv) {
    Options options = {0, 1024, 1024, 1024, 2, 10, -1, 0, 0, 14*GIB, 512, 1, 128, 16, 0, 1, 0, 0, 0, NULL};
    Context ctx = {0};
    int result = EXIT_FAILURE;
    // UCRT rejects a zero-sized line buffer; unbuffered output works for both builds.
    setvbuf(stdout, NULL, _IONBF, 0);
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--help")) { usage(argv[0]); return 0; }
        if (!strcmp(arg, "--info")) { options.info = 1; continue; }
        if (!strcmp(arg, "--vram-test")) {
            options.vram = 1;
            if (i+1 < argc && strncmp(argv[i+1], "--", 2)) {
                char *end;
                errno = 0;
                double gib = strtod(argv[++i], &end);
                if (errno || *end || !isfinite(gib) || gib <= 0 || gib > 1024) goto invalid;
                options.vram_bytes = (VkDeviceSize)(gib * (double)GIB) & ~UINT64_C(3);
                if (!options.vram_bytes) goto invalid;
            }
            continue;
        }
        if (i+1 >= argc) goto invalid;
        const char *value = argv[++i];
        if (!strcmp(arg, "--fp32-isa")) {
            options.fp32_isa = value;
        } else if (!strcmp(arg, "--type")) {
            options.type = -1;
            if (!strcmp(value, "all")) continue;
            for (int j = 0; j < 6; ++j) if (!strcmp(value, type_names[j])) options.type = j;
            if (options.type == -1) goto invalid;
        } else if (!strcmp(arg, "--fp32-kernel")) {
            if (!strcmp(value, "register")) options.fp32_register = 1;
            else if (!strcmp(value, "baseline")) options.fp32_register = 0;
            else goto invalid;
        } else {
            uint32_t *out;
            int allow_zero = 0;
            if (!strcmp(arg, "--device")) { out = &options.device; allow_zero = 1; }
            else if (!strcmp(arg, "--m")) out = &options.m;
            else if (!strcmp(arg, "--n")) out = &options.n;
            else if (!strcmp(arg, "--k")) out = &options.k;
            else if (!strcmp(arg, "--warmup")) { out = &options.warmup; allow_zero = 1; }
            else if (!strcmp(arg, "--iterations")) out = &options.iterations;
            else if (!strcmp(arg, "--vram-reserve-mib")) { out = &options.vram_reserve_mib; allow_zero = 1; }
            else if (!strcmp(arg, "--fp32-tile")) out = &options.fp32_tile;
            else if (!strcmp(arg, "--fp32-kstep")) out = &options.fp32_kstep;
            else if (!strcmp(arg, "--fp32-pad")) { out = &options.fp32_pad; allow_zero = 1; }
            else if (!strcmp(arg, "--fp32-prefetch")) { out = &options.fp32_prefetch; allow_zero = 1; }
            else if (!strcmp(arg, "--fp32-rows")) out = &options.fp32_rows;
            else if (!strcmp(arg, "--fp32-lds-prefetch")) { out = &options.fp32_lds_prefetch; allow_zero = 1; }
            else if (!strcmp(arg, "--fp32-pack-a")) { out = &options.fp32_pack_a; allow_zero = 1; }
            else goto invalid;
            if (!parse_uint(value, out, allow_zero)) goto invalid;
        }
    }
    if (options.iterations > 100000 || options.warmup > 100000 || options.vram_reserve_mib > 65536 ||
        (options.info && options.vram)) goto invalid;
    if ((options.fp32_tile != 64 && options.fp32_tile != 128) ||
        (options.fp32_kstep != 8 && options.fp32_kstep != 16 && options.fp32_kstep != 32) ||
        options.fp32_pad > 1 || options.fp32_prefetch > 1 || options.fp32_lds_prefetch > 1 || options.fp32_pack_a > 1) goto invalid;
    if (!options.fp32_rows) options.fp32_rows = options.fp32_tile;
    if (options.fp32_rows != 64 && options.fp32_rows != 128) goto invalid;
    if (init_context(&ctx, options.device, options.info)) {
        if (options.info || (options.vram ? vram_test(&ctx, options.vram_bytes,
            (VkDeviceSize)options.vram_reserve_mib << 20) : benchmark(&ctx, &options))) result = 0;
    }
    destroy_context(&ctx);
    return result;
invalid:
    fprintf(stderr, "Invalid arguments\n");
    usage(argv[0]);
    return 2;
}
