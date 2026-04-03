// SPDX-License-Identifier: Apache 2.0
// Copyright 2025 - Present, Light Transport Entertainment Inc.
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iterator>

#include <sys/stat.h>

#include "vulkan-runner.hh"

using namespace std;

namespace vl_cpp {

namespace vulkan {

// VulkanComputeRunner implementation

VulkanComputeRunner::VulkanComputeRunner() 
    : instance_(VK_NULL_HANDLE)
    , physicalDevice_(VK_NULL_HANDLE)
    , device_(VK_NULL_HANDLE)
    , computeQueue_(VK_NULL_HANDLE)
    , commandPool_(VK_NULL_HANDLE)
    , commandBuffer_(VK_NULL_HANDLE)
    , currentDeviceIndex_(0)
    , computeQueueFamilyIndex_(UINT32_MAX)
    , initialized_(false) {
}

VulkanComputeRunner::~VulkanComputeRunner() {
    cleanup();
}

bool VulkanComputeRunner::initialize(bool enableValidation) {
    if (initialized_) {
        return true;
    }

    if (!createInstance(enableValidation)) {
        return false;
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    
    if (deviceCount == 0) {
        setError("No Vulkan-capable devices found");
        return false;
    }

    physicalDevices_.resize(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, physicalDevices_.data());

    if (!selectPhysicalDevice(0)) {
        return false;
    }

    if (!createLogicalDevice()) {
        return false;
    }

    if (!createCommandPool()) {
        return false;
    }

    initialized_ = true;
    return true;
}

void VulkanComputeRunner::cleanup() {
    if (!initialized_) {
        return;
    }

    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);

        if (commandPool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, commandPool_, nullptr);
            commandPool_ = VK_NULL_HANDLE;
        }

        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }

    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }

    physicalDevice_ = VK_NULL_HANDLE;
    computeQueue_ = VK_NULL_HANDLE;
    commandBuffer_ = VK_NULL_HANDLE;
    initialized_ = false;
}

uint32_t VulkanComputeRunner::getDeviceCount() const {
    return static_cast<uint32_t>(physicalDevices_.size());
}

bool VulkanComputeRunner::selectDevice(uint32_t deviceIndex) {
    if (deviceIndex >= physicalDevices_.size()) {
        setError("Invalid device index");
        return false;
    }

    if (deviceIndex == currentDeviceIndex_ && device_ != VK_NULL_HANDLE) {
        return true;
    }

    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        vkDestroyDevice(device_, nullptr);
    }

    if (!selectPhysicalDevice(deviceIndex)) {
        return false;
    }

    if (!createLogicalDevice()) {
        return false;
    }

    if (!createCommandPool()) {
        return false;
    }

    currentDeviceIndex_ = deviceIndex;
    return true;
}

std::string VulkanComputeRunner::getDeviceName(uint32_t deviceIndex) const {
    if (deviceIndex >= physicalDevices_.size()) {
        return "Invalid device index";
    }

    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physicalDevices_[deviceIndex], &properties);
    return std::string(properties.deviceName);
}

bool VulkanComputeRunner::createBuffer(size_t size, VkBufferUsageFlags usage,
                                       VkMemoryPropertyFlags properties, BufferInfo& bufferInfo) {
    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = usage;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device_, &bufferCreateInfo, nullptr, &bufferInfo.buffer) != VK_SUCCESS) {
        setError("Failed to create buffer");
        return false;
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device_, bufferInfo.buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device_, &allocInfo, nullptr, &bufferInfo.memory) != VK_SUCCESS) {
        vkDestroyBuffer(device_, bufferInfo.buffer, nullptr);
        setError("Failed to allocate buffer memory");
        return false;
    }

    vkBindBufferMemory(device_, bufferInfo.buffer, bufferInfo.memory, 0);

    bufferInfo.size = size;
    bufferInfo.mappedPtr = nullptr;

    return true;
}

void VulkanComputeRunner::destroyBuffer(const BufferInfo& bufferInfo) {
    if (bufferInfo.mappedPtr != nullptr) {
        vkUnmapMemory(device_, bufferInfo.memory);
    }
    vkDestroyBuffer(device_, bufferInfo.buffer, nullptr);
    vkFreeMemory(device_, bufferInfo.memory, nullptr);
}

bool VulkanComputeRunner::createComputePipeline(const std::vector<uint32_t>& spirvCode,
                                                 const std::vector<VkDescriptorSetLayoutBinding>& bindings,
                                                 ComputePipeline& pipeline) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spirvCode.size() * sizeof(uint32_t);
    createInfo.pCode = spirvCode.data();

    if (vkCreateShaderModule(device_, &createInfo, nullptr, &pipeline.shaderModule) != VK_SUCCESS) {
        setError("Failed to create shader module");
        return false;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &pipeline.descriptorSetLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(device_, pipeline.shaderModule, nullptr);
        setError("Failed to create descriptor set layout");
        return false;
    }

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &pipeline.descriptorSetLayout;

    if (vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipeline.pipelineLayout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device_, pipeline.descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device_, pipeline.shaderModule, nullptr);
        setError("Failed to create pipeline layout");
        return false;
    }

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = pipeline.pipelineLayout;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = pipeline.shaderModule;
    pipelineInfo.stage.pName = "main";

    if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline.pipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(device_, pipeline.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device_, pipeline.descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device_, pipeline.shaderModule, nullptr);
        setError("Failed to create compute pipeline");
        return false;
    }

    std::vector<VkDescriptorPoolSize> poolSizes;
    poolSizes.resize(bindings.size());
    for (size_t i = 0; i < bindings.size(); i++) {
        poolSizes[i].type = bindings[i].descriptorType;
        poolSizes[i].descriptorCount = 1;
    }

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;

    if (vkCreateDescriptorPool(device_, &poolInfo, nullptr, &pipeline.descriptorPool) != VK_SUCCESS) {
        vkDestroyPipeline(device_, pipeline.pipeline, nullptr);
        vkDestroyPipelineLayout(device_, pipeline.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device_, pipeline.descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device_, pipeline.shaderModule, nullptr);
        setError("Failed to create descriptor pool");
        return false;
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = pipeline.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &pipeline.descriptorSetLayout;

    if (vkAllocateDescriptorSets(device_, &allocInfo, &pipeline.descriptorSet) != VK_SUCCESS) {
        vkDestroyDescriptorPool(device_, pipeline.descriptorPool, nullptr);
        vkDestroyPipeline(device_, pipeline.pipeline, nullptr);
        vkDestroyPipelineLayout(device_, pipeline.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device_, pipeline.descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device_, pipeline.shaderModule, nullptr);
        setError("Failed to allocate descriptor sets");
        return false;
    }

    return true;
}

void VulkanComputeRunner::destroyComputePipeline(const ComputePipeline& pipeline) {
    vkDestroyDescriptorPool(device_, pipeline.descriptorPool, nullptr);
    vkDestroyPipeline(device_, pipeline.pipeline, nullptr);
    vkDestroyPipelineLayout(device_, pipeline.pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device_, pipeline.descriptorSetLayout, nullptr);
    vkDestroyShaderModule(device_, pipeline.shaderModule, nullptr);
}

bool VulkanComputeRunner::createComputePipelineWithPushConstants(
    const std::vector<uint32_t>& spirvCode,
    const std::vector<VkDescriptorSetLayoutBinding>& bindings,
    uint32_t pushConstantSize,
    ComputePipeline& pipeline) {

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spirvCode.size() * sizeof(uint32_t);
    createInfo.pCode = spirvCode.data();

    if (vkCreateShaderModule(device_, &createInfo, nullptr, &pipeline.shaderModule) != VK_SUCCESS) {
        setError("Failed to create shader module");
        return false;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &pipeline.descriptorSetLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(device_, pipeline.shaderModule, nullptr);
        setError("Failed to create descriptor set layout");
        return false;
    }

    // Set up push constant range
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = pushConstantSize;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &pipeline.descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = (pushConstantSize > 0) ? 1 : 0;
    pipelineLayoutInfo.pPushConstantRanges = (pushConstantSize > 0) ? &pushConstantRange : nullptr;

    if (vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipeline.pipelineLayout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device_, pipeline.descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device_, pipeline.shaderModule, nullptr);
        setError("Failed to create pipeline layout");
        return false;
    }

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = pipeline.pipelineLayout;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = pipeline.shaderModule;
    pipelineInfo.stage.pName = "main";

    if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline.pipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(device_, pipeline.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device_, pipeline.descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device_, pipeline.shaderModule, nullptr);
        setError("Failed to create compute pipeline");
        return false;
    }

    std::vector<VkDescriptorPoolSize> poolSizes;
    poolSizes.resize(bindings.size());
    for (size_t i = 0; i < bindings.size(); i++) {
        poolSizes[i].type = bindings[i].descriptorType;
        poolSizes[i].descriptorCount = 1;
    }

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;

    if (vkCreateDescriptorPool(device_, &poolInfo, nullptr, &pipeline.descriptorPool) != VK_SUCCESS) {
        vkDestroyPipeline(device_, pipeline.pipeline, nullptr);
        vkDestroyPipelineLayout(device_, pipeline.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device_, pipeline.descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device_, pipeline.shaderModule, nullptr);
        setError("Failed to create descriptor pool");
        return false;
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = pipeline.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &pipeline.descriptorSetLayout;

    if (vkAllocateDescriptorSets(device_, &allocInfo, &pipeline.descriptorSet) != VK_SUCCESS) {
        vkDestroyDescriptorPool(device_, pipeline.descriptorPool, nullptr);
        vkDestroyPipeline(device_, pipeline.pipeline, nullptr);
        vkDestroyPipelineLayout(device_, pipeline.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device_, pipeline.descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device_, pipeline.shaderModule, nullptr);
        setError("Failed to allocate descriptor sets");
        return false;
    }

    return true;
}

bool VulkanComputeRunner::updateDescriptorSet(const ComputePipeline& pipeline,
                                               const std::vector<BufferInfo>& buffers) {
    std::vector<VkDescriptorBufferInfo> bufferInfos(buffers.size());
    std::vector<VkWriteDescriptorSet> descriptorWrites(buffers.size());

    for (size_t i = 0; i < buffers.size(); i++) {
        bufferInfos[i].buffer = buffers[i].buffer;
        bufferInfos[i].offset = 0;
        bufferInfos[i].range = buffers[i].size;

        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].pNext = nullptr;
        descriptorWrites[i].dstSet = pipeline.descriptorSet;
        descriptorWrites[i].dstBinding = static_cast<uint32_t>(i);
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
        descriptorWrites[i].pImageInfo = nullptr;
        descriptorWrites[i].pTexelBufferView = nullptr;
    }

    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(descriptorWrites.size()),
                          descriptorWrites.data(), 0, nullptr);
    return true;
}

bool VulkanComputeRunner::createDynamicDescriptorPool(uint32_t maxSets) {
    if (dynamicDescriptorPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, dynamicDescriptorPool_, nullptr);
    }
    // Use larger pool to be safe with driver quirks
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = maxSets * 8;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = maxSets;

    if (vkCreateDescriptorPool(device_, &poolInfo, nullptr, &dynamicDescriptorPool_) != VK_SUCCESS) {
        setError("Failed to create dynamic descriptor pool");
        return false;
    }

    // Pre-cache dynamic layouts for 1-5 bindings
    for (uint32_t n = 1; n <= 5; n++) getDynamicLayout(n);

    return true;
}

void VulkanComputeRunner::resetDynamicDescriptorPool() {
    if (dynamicDescriptorPool_ != VK_NULL_HANDLE) {
        vkResetDescriptorPool(device_, dynamicDescriptorPool_, 0);
    }
}

VkDescriptorSetLayout VulkanComputeRunner::getDynamicLayout(uint32_t nBindings) {
    if (nBindings == 0 || nBindings > 8) return VK_NULL_HANDLE;
    if (dynamicLayouts_[nBindings] != VK_NULL_HANDLE) return dynamicLayouts_[nBindings];

    std::vector<VkDescriptorSetLayoutBinding> bindings(nBindings);
    for (uint32_t i = 0; i < nBindings; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = nBindings;
    layoutInfo.pBindings = bindings.data();

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &layout) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    dynamicLayouts_[nBindings] = layout;
    return layout;
}

void VulkanComputeRunner::destroyDynamicLayouts() {
    for (int i = 0; i <= 8; i++) {
        if (dynamicLayouts_[i] != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device_, dynamicLayouts_[i], nullptr);
            dynamicLayouts_[i] = VK_NULL_HANDLE;
        }
    }
}

void VulkanComputeRunner::destroyDynamicDescriptorPool() {
    destroyDynamicLayouts();
    if (dynamicDescriptorPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, dynamicDescriptorPool_, nullptr);
        dynamicDescriptorPool_ = VK_NULL_HANDLE;
    }
}

VkDescriptorSet VulkanComputeRunner::allocateAndUpdateDescriptorSet(
    const ComputePipeline& pipeline, const std::vector<BufferInfo>& buffers) {
    VkDescriptorSet ds = VK_NULL_HANDLE;

    // Create a fresh layout matching buffer count (pipeline layouts don't work
    // with dynamic pool on some drivers like RADV)
    uint32_t nBindings = static_cast<uint32_t>(buffers.size());
    if (nBindings == 0 || nBindings > 8) {
        setError("Invalid binding count for dynamic descriptor set");
        return VK_NULL_HANDLE;
    }

    // Use cached layout by binding count
    VkDescriptorSetLayout layout = getDynamicLayout(nBindings);
    if (layout == VK_NULL_HANDLE) {
        setError("Failed to get dynamic layout");
        return VK_NULL_HANDLE;
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = dynamicDescriptorPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &layout;

    VkResult res = vkAllocateDescriptorSets(device_, &allocInfo, &ds);
    if (res != VK_SUCCESS) {
        setError("Failed to allocate dynamic descriptor set");
        return VK_NULL_HANDLE;
    }

    std::vector<VkDescriptorBufferInfo> bufferInfos(buffers.size());
    std::vector<VkWriteDescriptorSet> descriptorWrites(buffers.size());
    for (size_t i = 0; i < buffers.size(); i++) {
        bufferInfos[i].buffer = buffers[i].buffer;
        bufferInfos[i].offset = 0;
        bufferInfos[i].range = buffers[i].size;
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].pNext = nullptr;
        descriptorWrites[i].dstSet = ds;
        descriptorWrites[i].dstBinding = static_cast<uint32_t>(i);
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
        descriptorWrites[i].pImageInfo = nullptr;
        descriptorWrites[i].pTexelBufferView = nullptr;
    }
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(descriptorWrites.size()),
                          descriptorWrites.data(), 0, nullptr);
    return ds;
}

void VulkanComputeRunner::bindDescriptorSetDynamic(const ComputePipeline& pipeline, VkDescriptorSet ds) {
    vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline.pipelineLayout, 0, 1, &ds, 0, nullptr);
}

void VulkanComputeRunner::pushConstants(const ComputePipeline& pipeline, const void* data, uint32_t size) {
    vkCmdPushConstants(commandBuffer_, pipeline.pipelineLayout,
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, size, data);
}

bool VulkanComputeRunner::beginRecording() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool_;
    allocInfo.commandBufferCount = 1;

    vkAllocateCommandBuffers(device_, &allocInfo, &commandBuffer_);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(commandBuffer_, &beginInfo) != VK_SUCCESS) {
        setError("Failed to begin command buffer recording");
        return false;
    }

    return true;
}

void VulkanComputeRunner::bindComputePipeline(const ComputePipeline& pipeline) {
    vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
}

void VulkanComputeRunner::bindDescriptorSets(const ComputePipeline& pipeline) {
    vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, 
                           pipeline.pipelineLayout, 0, 1, &pipeline.descriptorSet, 0, nullptr);
}

void VulkanComputeRunner::dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) {
    vkCmdDispatch(commandBuffer_, groupCountX, groupCountY, groupCountZ);
}

void VulkanComputeRunner::computeBarrier() {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(commandBuffer_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
}

bool VulkanComputeRunner::endRecordingAndSubmit() {
    if (vkEndCommandBuffer(commandBuffer_) != VK_SUCCESS) {
        setError("Failed to end command buffer recording");
        return false;
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer_;

    if (vkQueueSubmit(computeQueue_, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        setError("Failed to submit command buffer");
        return false;
    }

    return true;
}

bool VulkanComputeRunner::waitForCompletion() {
    if (vkQueueWaitIdle(computeQueue_) != VK_SUCCESS) {
        setError("Failed to wait for queue completion");
        return false;
    }

    vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer_);
    commandBuffer_ = VK_NULL_HANDLE;
    return true;
}

bool VulkanComputeRunner::mapBuffer(const BufferInfo& bufferInfo, void** data) {
    if (vkMapMemory(device_, bufferInfo.memory, 0, bufferInfo.size, 0, data) != VK_SUCCESS) {
        setError("Failed to map buffer memory");
        return false;
    }
    return true;
}

void VulkanComputeRunner::unmapBuffer(const BufferInfo& bufferInfo) {
    vkUnmapMemory(device_, bufferInfo.memory);
}

bool VulkanComputeRunner::copyBuffer(const BufferInfo& srcBuffer, const BufferInfo& dstBuffer, size_t size) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool_;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device_, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer.buffer, dstBuffer.buffer, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(computeQueue_, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(computeQueue_);

    vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
    return true;
}

bool VulkanComputeRunner::createInstance(bool enableValidation) {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanComputeRunner";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "VisionLanguageCPP";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    std::vector<const char*> extensions;
    std::vector<const char*> validationLayers;

    if (enableValidation) {
        validationLayers.push_back("VK_LAYER_KHRONOS_validation");
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    if (vkCreateInstance(&createInfo, nullptr, &instance_) != VK_SUCCESS) {
        setError("Failed to create Vulkan instance");
        return false;
    }

    // Load instance-level Vulkan functions
    if (!vkewLoadInstance(instance_)) {
        setError(vkewGetError());
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
        return false;
    }

    return true;
}

bool VulkanComputeRunner::selectPhysicalDevice(uint32_t deviceIndex) {
    if (deviceIndex >= physicalDevices_.size()) {
        setError("Invalid device index");
        return false;
    }

    physicalDevice_ = physicalDevices_[deviceIndex];

    computeQueueFamilyIndex_ = findComputeQueueFamily(physicalDevice_);
    if (computeQueueFamilyIndex_ == UINT32_MAX) {
        setError("Failed to find compute queue family");
        return false;
    }

    return true;
}

bool VulkanComputeRunner::createLogicalDevice() {
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex_;
    queueCreateInfo.queueCount = 1;

    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceFeatures deviceFeatures{};

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = 0;
    createInfo.enabledLayerCount = 0;

    if (vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_) != VK_SUCCESS) {
        setError("Failed to create logical device");
        return false;
    }

    // Load device-level Vulkan functions
    if (!vkewLoadDevice(device_)) {
        setError(vkewGetError());
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
        return false;
    }

    vkGetDeviceQueue(device_, computeQueueFamilyIndex_, 0, &computeQueue_);
    return true;
}

bool VulkanComputeRunner::createCommandPool() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = computeQueueFamilyIndex_;

    if (vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_) != VK_SUCCESS) {
        setError("Failed to create command pool");
        return false;
    }

    return true;
}

uint32_t VulkanComputeRunner::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && 
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    return UINT32_MAX;
}

uint32_t VulkanComputeRunner::findComputeQueueFamily(VkPhysicalDevice device) {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            return i;
        }
    }

    return UINT32_MAX;
}

void VulkanComputeRunner::setError(const std::string& message) {
    lastError_ = message;
}

//  memory object.
struct _Memory {
  size_t size{};
  void *ptr{}; // @todo { Change type to uint64_t? }

  VkDeviceMemory memObjVK;
  int dummy;
};

//  module object.
struct _Module {
  VkShaderModule moduleObjVK;
  int dummy;
};

//  program object.
struct _Program {
  VkShaderModule progObjVK;
  int dummy;
};

//  kernel object.
struct _Kernel {
  VkPipeline kernObjVK;
  int dummy;
};

//  event object.
struct Event {
  VkEvent eventVK;
  int dummy;
};

vulkan::DeviceVulkan::DeviceVulkan(vulkan::DeviceTarget target) : vulkan::DeviceImpl() {
  assert(target == vulkan::vk_cpu || target == vulkan::vk_gpu || target == vulkan::vk_accel);

  if (target == vulkan::vk_cpu) {
    this->useCPU = true;
  } else {
    this->useCPU = false;
  }

  this->debug = false;
  this->measureProfile = false;

  this->instance = VK_NULL_HANDLE;
  this->devices.clear();
  this->commandQueues.clear();
}

vulkan::DeviceVulkan::~DeviceVulkan() {
  // Queues are automatically destroyed when the logical device is destroyed
  // vkDestroyQueue does not exist in Vulkan API
}

bool vulkan::DeviceVulkan::initialize(int reqPlatformID, int preferredDeviceID,
                               bool verbosity) {
  verb = verbosity;

  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Vulkan Runner";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  if (vkCreateInstance(&createInfo, nullptr, &this->instance) != VK_SUCCESS) {
    fprintf(stderr, "Failed to create Vulkan instance.\n");
    return false;
  }

  // Load instance-level Vulkan functions
  if (!vkewLoadInstance(this->instance)) {
    fprintf(stderr, "Failed to load Vulkan instance functions: %s\n", vkewGetError());
    return false;
  }

  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(this->instance, &deviceCount, nullptr);
  if (deviceCount == 0) {
    fprintf(stderr, "Failed to find GPUs with Vulkan support.\n");
    return false;
  }

  std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
  vkEnumeratePhysicalDevices(this->instance, &deviceCount, physicalDevices.data());

  this->devices.clear();
  for (const auto &device : physicalDevices) {
    this->devices.push_back(device);
  }

  this->currentDeviceID = 0;
  if (preferredDeviceID < (int)this->devices.size()) {
    this->currentDeviceID = preferredDeviceID;
  }

  return true;
}

int vulkan::DeviceVulkan::getNumDevices() {
  return int(this->devices.size());
}

int vulkan::DeviceVulkan::estimateMFlops(int deviceId) {
  // Placeholder implementation
  return 0;
}

bool vulkan::DeviceVulkan::shutdown() {
  vkDestroyInstance(this->instance, nullptr);
  return true;
}

vulkan::Program vulkan::DeviceVulkan::loadKernelSource(const char *filename, int nheaders,
                                       const char **headers, const char *options) {
  assert(this->instance != VK_NULL_HANDLE);

  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    fprintf(stderr, "Failed to open file: %s\n", filename);
    return nullptr;
  }

  size_t fileSize = (size_t)file.tellg();
  std::vector<char> buffer(fileSize);
  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();

  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = buffer.size();
  createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer.data());

  Program program = new _Program;
  // TODO: Fix this - devices array contains VkPhysicalDevice, we need VkDevice
  // For now, return a placeholder program
  // if (vkCreateShaderModule(logicalDevice, &createInfo, nullptr, &program->progObjVK) != VK_SUCCESS) {
    fprintf(stderr, "Shader module creation not yet implemented.\n");
    delete program;
    return nullptr;
  // }

  return program;
}

vulkan::Program vulkan::DeviceVulkan::loadKernelBinary(const char *filename) {
  return loadKernelSource(filename, 0, nullptr, nullptr);
}

bool vulkan::DeviceVulkan::getModule(vulkan::Program program, std::vector<char>& binary) {
  // Placeholder implementation
  return false;
}

vulkan::Memory vulkan::DeviceVulkan::alloc(vulkan::MemoryType memType, vulkan::MemoryAttrib memAttrib, size_t memSize) {
  assert(this->instance != VK_NULL_HANDLE);

  Memory mem = new _Memory;

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memSize;
  allocInfo.memoryTypeIndex = 0; // Placeholder

  // TODO: Fix this - devices array contains VkPhysicalDevice, we need VkDevice
  // Placeholder for now
  fprintf(stderr, "Memory allocation not yet implemented.\n");

  mem->size = memSize;
  mem->ptr = nullptr;

  return mem;
}

bool vulkan::DeviceVulkan::free(vulkan::Memory mem) {
  // TODO: Implement proper Vulkan memory deallocation
  delete mem;
  return true;
}

vulkan::Kernel vulkan::DeviceVulkan::createKernel(const vulkan::Program program, const char *functionName) {
  assert(this->instance != VK_NULL_HANDLE);

  Kernel kernel = new _Kernel;
  // Placeholder implementation
  kernel->kernObjVK = VK_NULL_HANDLE;

  return kernel;
}

bool vulkan::DeviceVulkan::bindMemoryObject(vulkan::Kernel kernel, int argNum, vulkan::Memory mem) {
  // Placeholder implementation
  return true;
}

bool vulkan::DeviceVulkan::setArg(vulkan::Kernel kernel, int argNum, size_t size, size_t align, void *arg) {
  // Placeholder implementation
  return true;
}

bool vulkan::DeviceVulkan::execute(int deviceID, vulkan::Kernel kernel, int dimension, size_t sizeX,
                           size_t sizeY, size_t sizeZ, size_t localSizeX, size_t localSizeY,
                           size_t localSizeZ) {
  // Placeholder implementation
  return true;
}

bool vulkan::DeviceVulkan::read(int deviceID, vulkan::Memory mem, size_t size, void *ptr) {
  // Placeholder implementation
  return true;
}

bool vulkan::DeviceVulkan::write(int deviceID, vulkan::Memory mem, size_t size, const void *ptr) {
  // Placeholder implementation
  return true;
}

bool InitializeVulkan() {
  // Initialize vkew runtime loader
  if (!vkewInit()) {
    fprintf(stderr, "Failed to initialize vkew: %s\n", vkewGetError());
    return false;
  }
  return true;
}

// Device wrapper implementations
Device::Device(DeviceTarget target) {
  impl = new DeviceVulkan(target);
}

Device::~Device() {
  delete impl;
}

bool Device::initialize(int platformID, int preferredDeviceID, bool verbosity) {
  return impl->initialize(platformID, preferredDeviceID, verbosity);
}

int Device::getNumDevices() {
  return impl->getNumDevices();
}

int Device::estimateMFlops(int deviceId) {
  return impl->estimateMFlops(deviceId);
}

Program Device::loadKernelSource(const char *filename, int nheaders,
                                 const char **headers, const char *options) {
  return impl->loadKernelSource(filename, nheaders, headers, options);
}

Program Device::loadKernelBinary(const char *filename) {
  return impl->loadKernelBinary(filename);
}

Kernel Device::createKernel(const Program program, const char *functionName) {
  return impl->createKernel(program, functionName);
}

bool Device::getModule(Program program, std::vector<char>& binary) {
  return impl->getModule(program, binary);
}

Memory Device::alloc(MemoryType memType, MemoryAttrib memAttrib, size_t memSize) {
  return impl->alloc(memType, memAttrib, memSize);
}

bool Device::free(Memory mem) {
  return impl->free(mem);
}

bool Device::write(int deviceID, Memory mem, size_t size, const void *ptr) {
  return impl->write(deviceID, mem, size, ptr);
}

bool Device::read(int deviceID, Memory mem, size_t size, void *ptr) {
  return impl->read(deviceID, mem, size, ptr);
}

bool Device::bindMemoryObject(Kernel kernel, int argNum, Memory mem) {
  return impl->bindMemoryObject(kernel, argNum, mem);
}

bool Device::setArg(Kernel kernel, int argNum, size_t size, size_t align, void *arg) {
  return impl->setArg(kernel, argNum, size, align, arg);
}

bool Device::execute(int deviceID, Kernel kernel, int dimension,
                     size_t sizeX, size_t sizeY, size_t sizeZ,
                     size_t localSizeX, size_t localSizeY, size_t localSizeZ) {
  return impl->execute(deviceID, kernel, dimension, sizeX, sizeY, sizeZ,
                       localSizeX, localSizeY, localSizeZ);
}

bool Device::shutdown() {
  return impl->shutdown();
}

void Device::PushError(const std::string &msg) {
  impl->PushError(msg);
}

// DeviceImpl base class implementation
DeviceImpl::DeviceImpl() : useCPU(false), debug(false), measureProfile(false), verb(false), currentDeviceID(0) {
}

DeviceImpl::~DeviceImpl() {
}

} // namespace vulkan

} // namespace vl_cpp
