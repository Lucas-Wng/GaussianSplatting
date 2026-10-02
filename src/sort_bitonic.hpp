#pragma once

// GPU bitonic depth sort: in-place, one dispatch per (k, j) stage over a key/value array
// padded to a power of two. See shaders/compute.slang for the compute side.

#include "ply_loader.hpp"
#include "vk_utils.hpp"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <vector>

// Must match SortPush in shaders/compute.slang.
struct SortPushConstants
{
	glm::vec4 viewRow2;        // z = dot(viewRow2.xyz, pos) + viewRow2.w
	uint32_t  splatCount;
	uint32_t  paddedCount;
	uint32_t  j;               // bitonic partner stride
	uint32_t  k;               // bitonic subsequence size
};

class BitonicSorter
{
  public:
	// indexBuffers: per-frame draw-index buffers owned by the caller (shared with the CPU
	// sort path and the graphics pipeline's descriptor set) -- this sort writes into them
	// in place, as its "values" array, rather than owning a copy.
	void create(const vk::raii::Device &device, const vk::raii::PhysicalDevice &physicalDevice,
	           const std::string &computeShaderSpv, vk::raii::Buffer &splatBuffer, uint32_t splatCount,
	           uint32_t paddedCount, std::vector<vk::raii::Buffer> &indexBuffers)
	{
		int framesInFlight = static_cast<int>(indexBuffers.size());

		std::array bindings{
		    // binding 0: splat data (read) — positions for depth keys
		    vk::DescriptorSetLayoutBinding{
		        .binding = 0, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
		    // binding 1: sort keys (read/write scratch)
		    vk::DescriptorSetLayoutBinding{
		        .binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
		    // binding 2: sort values == draw indices (read/write, sorted in place)
		    vk::DescriptorSetLayoutBinding{
		        .binding = 2, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute}};
		vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()};
		descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

		vk::raii::ShaderModule shaderModule = vkutil::createShaderModule(device, vkutil::readFile(computeShaderSpv));

		vk::PushConstantRange pushRange{.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(SortPushConstants)};
		vk::PipelineLayoutCreateInfo plInfo{
		    .setLayoutCount = 1, .pSetLayouts = &*descriptorSetLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushRange};
		pipelineLayout = vk::raii::PipelineLayout(device, plInfo);

		auto makePipeline = [&](const char *entry) {
			vk::PipelineShaderStageCreateInfo stage{.stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = entry};
			vk::ComputePipelineCreateInfo     info{.stage = stage, .layout = pipelineLayout};
			return vk::raii::Pipeline(device, nullptr, info);
		};
		initPipeline    = makePipeline("sortInitMain");
		bitonicPipeline = makePipeline("sortBitonicMain");

		// Per-frame device-local scratch holding the sort's depth keys.
		vk::DeviceSize keyBufferSize = sizeof(uint32_t) * paddedCount;
		for (int i = 0; i < framesInFlight; i++)
		{
			auto [buffer, bufferMem] = vkutil::createBuffer(device, physicalDevice, keyBufferSize, vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
			sortKeyBuffers.emplace_back(std::move(buffer));
			sortKeyBuffersMemory.emplace_back(std::move(bufferMem));
		}

		std::array poolSizes{vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 3u * static_cast<uint32_t>(framesInFlight)}};
		descriptorPool = vk::raii::DescriptorPool(device, {.flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		                                                   .maxSets       = static_cast<uint32_t>(framesInFlight),
		                                                   .poolSizeCount = 1,
		                                                   .pPoolSizes    = poolSizes.data()});

		std::vector<vk::DescriptorSetLayout> layouts(framesInFlight, *descriptorSetLayout);
		vk::DescriptorSetAllocateInfo        allocInfo{.descriptorPool = descriptorPool, .descriptorSetCount = static_cast<uint32_t>(layouts.size()), .pSetLayouts = layouts.data()};
		descriptorSets = device.allocateDescriptorSets(allocInfo);

		for (int i = 0; i < framesInFlight; i++)
		{
			vk::DescriptorBufferInfo splatInfo{.buffer = splatBuffer, .offset = 0, .range = sizeof(GpuSplat) * splatCount};
			vk::DescriptorBufferInfo keyInfo{.buffer = sortKeyBuffers[i], .offset = 0, .range = sizeof(uint32_t) * paddedCount};
			vk::DescriptorBufferInfo valueInfo{.buffer = indexBuffers[i], .offset = 0, .range = sizeof(uint32_t) * paddedCount};

			std::array descriptorWrites{
			    vk::WriteDescriptorSet{.dstSet = descriptorSets[i], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &splatInfo},
			    vk::WriteDescriptorSet{.dstSet = descriptorSets[i], .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &keyInfo},
			    vk::WriteDescriptorSet{.dstSet = descriptorSets[i], .dstBinding = 2, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &valueInfo}};
			device.updateDescriptorSets(descriptorWrites, {});
		}
	}

	// Records the sort in place into indexBuffers[frameIndex] (bound as this frame's
	// descriptor set's "values" buffer back in create()).
	void record(vk::raii::CommandBuffer &cmd, uint32_t frameIndex, glm::vec4 viewRow2, uint32_t splatCount, uint32_t paddedCount) const
	{
		SortPushConstants pc{.viewRow2 = viewRow2, .splatCount = splatCount, .paddedCount = paddedCount, .j = 0, .k = 0};

		// Serializes dependent dispatches (write -> read/write).
		const vk::MemoryBarrier2 computeBarrier{
		    .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
		    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
		    .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
		    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite};
		const vk::DependencyInfo computeDependency{.memoryBarrierCount = 1, .pMemoryBarriers = &computeBarrier};

		cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, *descriptorSets[frameIndex], nullptr);

		// 1) Fill keys (order-preserving depth) and identity values for every padded slot.
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *initPipeline);
		cmd.pushConstants<SortPushConstants>(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pc);
		uint32_t groups = (paddedCount + 255u) / 256u;
		cmd.dispatch(groups, 1, 1);
		cmd.pipelineBarrier2(computeDependency);

		// 2) Bitonic network: one dispatch per (k, j) stage, in place on keys + values.
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *bitonicPipeline);
		for (uint32_t k = 2; k <= paddedCount; k <<= 1)
		{
			for (uint32_t j = k >> 1; j > 0; j >>= 1)
			{
				pc.k = k;
				pc.j = j;
				cmd.pushConstants<SortPushConstants>(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pc);
				cmd.dispatch(groups, 1, 1);
				cmd.pipelineBarrier2(computeDependency);
			}
		}

		// 3) Hand the sorted indices to the vertex stage.
		const vk::MemoryBarrier2 toVertexBarrier{
		    .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
		    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
		    .dstStageMask  = vk::PipelineStageFlagBits2::eVertexShader,
		    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead};
		cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &toVertexBarrier});
	}

  private:
	vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
	vk::raii::PipelineLayout      pipelineLayout      = nullptr;
	vk::raii::Pipeline            initPipeline        = nullptr;
	vk::raii::Pipeline            bitonicPipeline     = nullptr;

	std::vector<vk::raii::Buffer>       sortKeyBuffers;
	std::vector<vk::raii::DeviceMemory> sortKeyBuffersMemory;

	// Declared (and so destructed) *after* descriptorSets below it, members destruct in
	// reverse declaration order, so the pool outlives the sets it needs to free them.
	vk::raii::DescriptorPool             descriptorPool = nullptr;
	std::vector<vk::raii::DescriptorSet> descriptorSets;
};
