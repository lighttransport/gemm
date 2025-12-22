//
// Copyright 2025 - Present, Light Transport Entertainment Inc.
//
#pragma once

#include <cstdlib>
#include <vector>
#include <map>
#include <string>

#include "vkew.h"

namespace vl_cpp {

namespace vulkan {

typedef enum {
  vk_cpu = 1,
  vk_gpu,
  vk_accel,
} DeviceTarget;

typedef enum {
  host = 1,
  device_global,
  device_cached_global,
  device_constant,
  device_texture,
} MemoryType;

typedef enum {
  ro, // read only.
  wo, // write only.
  rw, // read and write.
} MemoryAttrib;

// Forward decl.
struct _Memory;
typedef struct _Memory *Memory;
struct _Program;
typedef struct _Program *Program;
struct _Kernel;
typedef struct _Kernel *Kernel;
class DeviceImpl;

bool InitializeVulkan();

// Base class of Vulkan device.
class Device {
public:
  Device(DeviceTarget target);
  ~Device();

  // Function: initialize
  // Initializes Vulkan device.
  bool initialize(int platformID = 0, int preferredDeviceID = 0,
                  bool verbosity = false);

  // Function: getNumDevices
  // Returns # of available devices.
  int getNumDevices();

  // Function: estimateMFlops
  // Returns the Mflops of ith device.
  int estimateMFlops(int deviceId);

  // Function: loadKernelSource
  // Loads and compiles compute shader from SPIR-V source file.
  Program loadKernelSource(const char *filename, int nheaders = 0,
                           const char **headers = nullptr, const char *options = nullptr);

  // Function: loadKernelBinary
  // Loads precompiled SPIR-V binary from the file.
  Program loadKernelBinary(const char *filename);

  Kernel createKernel(const Program program, const char *functionName);

  // Function getModule
  // Get compiled SPIR-V kernel module.
  bool getModule(Program program, std::vector<char>& binary);

  Memory alloc(MemoryType memType, MemoryAttrib memAttrib,
               size_t memSize);

  bool free(Memory mem);

  bool write(int deviceID, Memory mem, size_t size, const void *ptr);
  bool read(int deviceID, Memory mem, size_t size, void *ptr);

  bool bindMemoryObject(Kernel kernel, int argNum, Memory mem);
  bool setArg(Kernel kernel, int argNum, size_t size, size_t align,
              void *arg);

  bool execute(int deviceID, Kernel kernel, int dimension,
               size_t sizeX, size_t sizeY = 1, size_t sizeZ = 1,
               size_t localSizeX = 1, size_t localSizeY = 1, size_t localSizeZ = 1);

  bool shutdown();

  void PushError(const std::string &msg);

private:
  DeviceImpl *impl;
};

// Implementation class
class DeviceImpl {
public:
  DeviceImpl();
  virtual ~DeviceImpl();

  virtual bool initialize(int reqPlatformID, int preferredDeviceID,
                          bool verbosity) = 0;
  virtual int getNumDevices() = 0;
  virtual int estimateMFlops(int deviceId) = 0;
  virtual Program loadKernelSource(const char *filename, int nheaders,
                                   const char **headers, const char *options) = 0;
  virtual Program loadKernelBinary(const char *filename) = 0;
  virtual bool getModule(Program program, std::vector<char>& binary) = 0;
  virtual Memory alloc(MemoryType memType, MemoryAttrib memAttrib,
                       size_t memSize) = 0;
  virtual bool free(Memory mem) = 0;
  virtual Kernel createKernel(const Program program, const char *functionName) = 0;
  virtual bool bindMemoryObject(Kernel kernel, int argNum, Memory mem) = 0;
  virtual bool setArg(Kernel kernel, int argNum, size_t size, size_t align,
                      void *arg) = 0;
  virtual bool execute(int deviceID, Kernel kernel, int dimension,
                       size_t sizeX, size_t sizeY, size_t sizeZ,
                       size_t localSizeX, size_t localSizeY, size_t localSizeZ) = 0;
  virtual bool read(int deviceID, Memory mem, size_t size, void *ptr) = 0;
  virtual bool write(int deviceID, Memory mem, size_t size, const void *ptr) = 0;
  virtual bool shutdown() = 0;

  void PushError(const std::string &msg) {
    err_ += msg;
  }

  std::string GetError() const {
    return err_;
  }

protected:
  std::string err_;
  bool useCPU;
  bool debug;
  bool measureProfile;
  bool verb;
  int currentDeviceID;
};

// Vulkan-specific device implementation
class DeviceVulkan : public DeviceImpl {
public:
  DeviceVulkan(DeviceTarget target);
  ~DeviceVulkan();

  bool initialize(int reqPlatformID, int preferredDeviceID,
                  bool verbosity) override;
  int getNumDevices() override;
  int estimateMFlops(int deviceId) override;
  Program loadKernelSource(const char *filename, int nheaders,
                           const char **headers, const char *options) override;
  Program loadKernelBinary(const char *filename) override;
  bool getModule(Program program, std::vector<char>& binary) override;
  Memory alloc(MemoryType memType, MemoryAttrib memAttrib,
               size_t memSize) override;
  bool free(Memory mem) override;
  Kernel createKernel(const Program program, const char *functionName) override;
  bool bindMemoryObject(Kernel kernel, int argNum, Memory mem) override;
  bool setArg(Kernel kernel, int argNum, size_t size, size_t align,
              void *arg) override;
  bool execute(int deviceID, Kernel kernel, int dimension,
               size_t sizeX, size_t sizeY, size_t sizeZ,
               size_t localSizeX, size_t localSizeY, size_t localSizeZ) override;
  bool read(int deviceID, Memory mem, size_t size, void *ptr) override;
  bool write(int deviceID, Memory mem, size_t size, const void *ptr) override;
  bool shutdown() override;

private:
  VkInstance instance;
  std::vector<VkPhysicalDevice> devices;
  std::vector<VkDevice> logicalDevices;
  std::vector<VkQueue> commandQueues;
  std::vector<VkCommandPool> commandPools;
  std::vector<uint32_t> queueFamilyIndices;

  bool createLogicalDevice(int deviceIndex);
  bool findQueueFamilies(VkPhysicalDevice device, uint32_t& computeQueueFamily);
  uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
};

// VulkanComputeRunner - High-level compute runner interface
class VulkanComputeRunner {
public:
  VulkanComputeRunner();
  ~VulkanComputeRunner();

  // Initialize Vulkan compute environment
  bool initialize(bool enableValidation = false);
  
  // Cleanup resources
  void cleanup();

  // Device management
  uint32_t getDeviceCount() const;
  bool selectDevice(uint32_t deviceIndex);
  std::string getDeviceName(uint32_t deviceIndex) const;
  
  // Buffer management
  struct BufferInfo {
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t size;
    void* mappedPtr;
  };
  
  bool createBuffer(size_t size, VkBufferUsageFlags usage, 
                    VkMemoryPropertyFlags properties, BufferInfo& bufferInfo);
  void destroyBuffer(const BufferInfo& bufferInfo);
  
  // Compute pipeline management
  struct ComputePipeline {
    VkShaderModule shaderModule;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
  };
  
  bool createComputePipeline(const std::vector<uint32_t>& spirvCode,
                             const std::vector<VkDescriptorSetLayoutBinding>& bindings,
                             ComputePipeline& pipeline);
  // Create pipeline with push constant support
  bool createComputePipelineWithPushConstants(const std::vector<uint32_t>& spirvCode,
                             const std::vector<VkDescriptorSetLayoutBinding>& bindings,
                             uint32_t pushConstantSize,
                             ComputePipeline& pipeline);
  void destroyComputePipeline(const ComputePipeline& pipeline);

  // Descriptor set management
  bool updateDescriptorSet(const ComputePipeline& pipeline,
                           const std::vector<BufferInfo>& buffers);

  // Command buffer management
  bool beginRecording();
  void bindComputePipeline(const ComputePipeline& pipeline);
  void bindDescriptorSets(const ComputePipeline& pipeline);
  void pushConstants(const ComputePipeline& pipeline, const void* data, uint32_t size);
  void dispatch(uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1);
  bool endRecordingAndSubmit();
  
  // Synchronization
  bool waitForCompletion();
  
  // Utility functions
  bool mapBuffer(const BufferInfo& bufferInfo, void** data);
  void unmapBuffer(const BufferInfo& bufferInfo);
  bool copyBuffer(const BufferInfo& srcBuffer, const BufferInfo& dstBuffer, size_t size);
  
  // Error handling
  std::string getLastError() const { return lastError_; }

private:
  VkInstance instance_;
  VkPhysicalDevice physicalDevice_;
  VkDevice device_;
  VkQueue computeQueue_;
  VkCommandPool commandPool_;
  VkCommandBuffer commandBuffer_;
  
  uint32_t currentDeviceIndex_;
  uint32_t computeQueueFamilyIndex_;
  std::vector<VkPhysicalDevice> physicalDevices_;
  
  std::string lastError_;
  bool initialized_;
  
  bool createInstance(bool enableValidation);
  bool selectPhysicalDevice(uint32_t deviceIndex);
  bool createLogicalDevice();
  bool createCommandPool();
  uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
  uint32_t findComputeQueueFamily(VkPhysicalDevice device);
  
  void setError(const std::string& message);
};

} // namespace vulkan

} // namespace vl_cpp