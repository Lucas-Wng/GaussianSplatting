#pragma once

// Small, stateless Vulkan helpers shared by Application (main.cpp) and the sort backends
// (sort_bitonic.hpp, sort_radix.hpp). Everything here takes the device/queue/pool it needs
// as explicit parameters rather than storing them, so callers don't have to worry about
// extra cross-object lifetime/destruction-order dependencies.

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vkutil
{
	inline uint32_t findMemoryType(const vk::raii::PhysicalDevice &physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties)
	{
		vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
		{
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
				return i;
		}
		throw std::runtime_error("failed to find suitable memory type!");
	}

	inline std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(
	    const vk::raii::Device &device, const vk::raii::PhysicalDevice &physicalDevice,
	    vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties)
	{
		vk::BufferCreateInfo   bufferInfo{.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive};
		vk::raii::Buffer       buffer          = vk::raii::Buffer(device, bufferInfo);
		vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
		vk::MemoryAllocateInfo allocInfo{.allocationSize = memRequirements.size, .memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties)};
		vk::raii::DeviceMemory bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
		buffer.bindMemory(*bufferMemory, 0);
		return {std::move(buffer), std::move(bufferMemory)};
	}

	inline void copyBuffer(const vk::raii::Device &device, const vk::raii::CommandPool &commandPool, const vk::raii::Queue &queue,
	                       const vk::raii::Buffer &srcBuffer, const vk::raii::Buffer &dstBuffer, vk::DeviceSize size)
	{
		vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};
		vk::raii::CommandBuffer       commandCopyBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());
		commandCopyBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy{.srcOffset = 0, .dstOffset = 0, .size = size});
		commandCopyBuffer.end();
		queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandCopyBuffer}, nullptr);
		queue.waitIdle();
	}

	inline void uploadToDevice(const vk::raii::Device &device, const vk::raii::PhysicalDevice &physicalDevice,
	                          const vk::raii::CommandPool &commandPool, const vk::raii::Queue &queue,
	                          const vk::raii::Buffer &dst, const void *src, vk::DeviceSize size)
	{
		auto [staging, stagingMem] =
		    createBuffer(device, physicalDevice, size, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
		void *p = stagingMem.mapMemory(0, size);
		memcpy(p, src, size);
		stagingMem.unmapMemory();
		copyBuffer(device, commandPool, queue, staging, dst, size);
	}

	inline void readbackFromDevice(const vk::raii::Device &device, const vk::raii::PhysicalDevice &physicalDevice,
	                               const vk::raii::CommandPool &commandPool, const vk::raii::Queue &queue,
	                               const vk::raii::Buffer &src, void *dst, vk::DeviceSize size)
	{
		auto [staging, stagingMem] =
		    createBuffer(device, physicalDevice, size, vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
		copyBuffer(device, commandPool, queue, src, staging, size);
		void *p = stagingMem.mapMemory(0, size);
		memcpy(dst, p, size);
		stagingMem.unmapMemory();
	}

	// Global compute->compute barrier (storage write -> read/write) between dependent dispatches.
	inline void recordComputeBarrier(const vk::raii::CommandBuffer &cmd)
	{
		vk::MemoryBarrier2 b{
		    .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
		    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
		    .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
		    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite};
		cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &b});
	}

	inline std::vector<char> readFile(const std::string &filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);
		if (!file.is_open())
			throw std::runtime_error("failed to open file: " + filename);
		std::vector<char> buffer(file.tellg());
		file.seekg(0, std::ios::beg);
		file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
		return buffer;
	}

	inline vk::raii::ShaderModule createShaderModule(const vk::raii::Device &device, const std::vector<char> &code)
	{
		vk::ShaderModuleCreateInfo createInfo{.codeSize = code.size(), .pCode = reinterpret_cast<const uint32_t *>(code.data())};
		return vk::raii::ShaderModule(device, createInfo);
	}
}        // namespace vkutil
