#pragma once

// GPU LSD radix depth sort: an 8-bit-digit, stable radix sort (key = uint2, value = uint)
// plus the multi-level exclusive scan it needs, used to produce back-to-front draw indices.
// See shaders/radix.slang for the compute side; the Splat struct there must stay in sync
// with GpuSplat in ply_loader.hpp (only `position` is read, but the stride must still match).

#include "ply_loader.hpp"
#include "vk_utils.hpp"

#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

// Must match RadixInitPush in shaders/radix.slang.
struct RadixInitPushConstants
{
	glm::vec4 viewRow2;        // z = dot(viewRow2.xyz, pos) + viewRow2.w
	uint32_t  splatCount;
	uint32_t  paddedCount;
};

// Must match RadixPush in shaders/radix.slang.
struct RadixPushConstants
{
	uint32_t count;            // active elements (radix) / scan length
	uint32_t digit;            // 0..7: which 8-bit digit of the 64-bit key
	uint32_t numBlocks;        // ceil(count / 256)
};

class RadixSorter
{
  public:
	// Number of 8-bit digit passes needed to fully sort the low 32 bits of the depth key.
	// The key is packed as uint2(lo, hi) for a future 64-bit tile+depth key, but the
	// depth-only key used by record() only ever puts information in the low word (hi is 0
	// for real splats, and padding already sorts last on the low word alone via
	// 0xFFFFFFFF) -- 4 passes is sufficient and halves this sort's cost versus the full
	// 8-pass/64-bit width selfTest() validates.
	static constexpr uint32_t kDepthKeyPasses = 4;

	// radixShaderSpv: path to the compiled radix.spv. framesInFlight: ping-pong buffers are
	// duplicated per frame in flight so frame N+1's dispatches never race frame N's
	// still-running GPU work on the same scratch buffers.
	void create(const vk::raii::Device &device, const vk::raii::PhysicalDevice &physicalDevice,
	           const std::string &radixShaderSpv, vk::raii::Buffer &splatBuffer, uint32_t splatCount,
	           uint32_t paddedCount, int framesInFlight)
	{
		radixCapacity  = std::max(256u, paddedCount);        // reuse the per-splat padded count
		radixNumBlocks = radixCapacity / 256u;

		auto makeDeviceBuffer = [&](vk::DeviceSize size) {
			return vkutil::createBuffer(device, physicalDevice, size,
			                            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
			                            vk::MemoryPropertyFlagBits::eDeviceLocal);
		};

		// Descriptor set layouts: radix (bindings 0-4), scan (bindings 5-6).
		{
			std::array<vk::DescriptorSetLayoutBinding, 5> b{};
			for (uint32_t i = 0; i < b.size(); ++i)
				b[i] = {.binding = i, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute};
			radixSetLayout = vk::raii::DescriptorSetLayout(device, {.bindingCount = static_cast<uint32_t>(b.size()), .pBindings = b.data()});
		}
		{
			std::array b{
			    vk::DescriptorSetLayoutBinding{.binding = 5, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
			    vk::DescriptorSetLayoutBinding{.binding = 6, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute}};
			scanSetLayout = vk::raii::DescriptorSetLayout(device, {.bindingCount = static_cast<uint32_t>(b.size()), .pBindings = b.data()});
		}

		vk::PushConstantRange pcRange{.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(RadixPushConstants)};
		radixPipelineLayout = vk::raii::PipelineLayout(device, {.setLayoutCount = 1, .pSetLayouts = &*radixSetLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcRange});
		scanPipelineLayout  = vk::raii::PipelineLayout(device, {.setLayoutCount = 1, .pSetLayouts = &*scanSetLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcRange});

		// Depth-key init: splats(7) -> keysOut(8)/valsOut(9), own layout/set.
		{
			std::array b{
			    vk::DescriptorSetLayoutBinding{.binding = 7, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
			    vk::DescriptorSetLayoutBinding{.binding = 8, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
			    vk::DescriptorSetLayoutBinding{.binding = 9, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute}};
			radixInitSetLayout = vk::raii::DescriptorSetLayout(device, {.bindingCount = static_cast<uint32_t>(b.size()), .pBindings = b.data()});
		}
		vk::PushConstantRange initPcRange{.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(RadixInitPushConstants)};
		radixInitPipelineLayout =
		    vk::raii::PipelineLayout(device, {.setLayoutCount = 1, .pSetLayouts = &*radixInitSetLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &initPcRange});

		vk::raii::ShaderModule mod      = vkutil::createShaderModule(device, vkutil::readFile(radixShaderSpv));
		auto                   makePipe = [&](const char *entry, vk::raii::PipelineLayout &layout) {
            vk::PipelineShaderStageCreateInfo stage{.stage = vk::ShaderStageFlagBits::eCompute, .module = mod, .pName = entry};
            return vk::raii::Pipeline(device, nullptr, vk::ComputePipelineCreateInfo{.stage = stage, .layout = layout});
		};
		radixHistogramPipeline = makePipe("radixHistogramMain", radixPipelineLayout);
		radixScatterPipeline   = makePipe("radixScatterMain", radixPipelineLayout);
		scanBlockPipeline      = makePipe("scanBlockMain", scanPipelineLayout);
		scanAddPipeline        = makePipe("scanAddMain", scanPipelineLayout);
		radixInitPipeline      = makePipe("radixDepthInitMain", radixInitPipelineLayout);

		// Scan levels (lengths) over the block histogram are the same shape for every frame.
		std::vector<uint32_t> scanLens;
		std::vector<uint32_t> scanBlocks;
		for (uint32_t len = radixCapacity;;)
		{
			uint32_t nb = (len + 255u) / 256u;
			scanLens.push_back(len);
			scanBlocks.push_back(nb);
			if (nb <= 1u)
				break;
			len = nb;
		}
		uint32_t numScan = static_cast<uint32_t>(scanLens.size());

		std::array poolSizes{vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = (13u + 2u * numScan) * static_cast<uint32_t>(framesInFlight)}};
		radixDescriptorPool = vk::raii::DescriptorPool(device, {.flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		                                                        .maxSets       = (3u + numScan) * static_cast<uint32_t>(framesInFlight),
		                                                        .poolSizeCount = 1,
		                                                        .pPoolSizes    = poolSizes.data()});

		auto allocOne = [&](vk::raii::DescriptorSetLayout &layout) {
			vk::DescriptorSetAllocateInfo ai{.descriptorPool = radixDescriptorPool, .descriptorSetCount = 1, .pSetLayouts = &*layout};
			return std::move(device.allocateDescriptorSets(ai).front());
		};

		radixFrames.clear();
		for (int f = 0; f < framesInFlight; ++f)
		{
			RadixFrame rf;
			std::tie(rf.keyBufferA, rf.keyBufferAMem)           = makeDeviceBuffer(sizeof(uint64_t) * radixCapacity);
			std::tie(rf.keyBufferB, rf.keyBufferBMem)           = makeDeviceBuffer(sizeof(uint64_t) * radixCapacity);
			std::tie(rf.valBufferA, rf.valBufferAMem)           = makeDeviceBuffer(sizeof(uint32_t) * radixCapacity);
			std::tie(rf.valBufferB, rf.valBufferBMem)           = makeDeviceBuffer(sizeof(uint32_t) * radixCapacity);
			std::tie(rf.blockHistBuffer, rf.blockHistBufferMem) = makeDeviceBuffer(sizeof(uint32_t) * radixCapacity);

			for (uint32_t k = 0; k < numScan; ++k)
			{
				ScanLevel lvl;
				lvl.dataLen                     = scanLens[k];
				lvl.numBlocks                   = scanBlocks[k];
				std::tie(lvl.sums, lvl.sumsMem) = makeDeviceBuffer(sizeof(uint32_t) * lvl.numBlocks);
				rf.scanLevels.push_back(std::move(lvl));
			}

			rf.radixSetAB = allocOne(radixSetLayout);
			rf.radixSetBA = allocOne(radixSetLayout);
			writeRadixSet(device, rf.radixSetAB, rf.keyBufferA, rf.valBufferA, rf.keyBufferB, rf.valBufferB, rf.blockHistBuffer);
			writeRadixSet(device, rf.radixSetBA, rf.keyBufferB, rf.valBufferB, rf.keyBufferA, rf.valBufferA, rf.blockHistBuffer);

			rf.radixInitSet = allocOne(radixInitSetLayout);
			{
				vk::DescriptorBufferInfo splatInfo{.buffer = splatBuffer, .offset = 0, .range = sizeof(GpuSplat) * splatCount};
				vk::DescriptorBufferInfo koutInfo{.buffer = rf.keyBufferA, .offset = 0, .range = sizeof(uint64_t) * radixCapacity};
				vk::DescriptorBufferInfo voutInfo{.buffer = rf.valBufferA, .offset = 0, .range = sizeof(uint32_t) * radixCapacity};
				std::array w{
				    vk::WriteDescriptorSet{.dstSet = rf.radixInitSet, .dstBinding = 7, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &splatInfo},
				    vk::WriteDescriptorSet{.dstSet = rf.radixInitSet, .dstBinding = 8, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &koutInfo},
				    vk::WriteDescriptorSet{.dstSet = rf.radixInitSet, .dstBinding = 9, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &voutInfo}};
				device.updateDescriptorSets(w, {});
			}

			for (uint32_t k = 0; k < numScan; ++k)
			{
				rf.scanSets.push_back(allocOne(scanSetLayout));
				vk::raii::Buffer &data = (k == 0) ? rf.blockHistBuffer : rf.scanLevels[k - 1].sums;
				writeScanSet(device, rf.scanSets[k], data, rf.scanLevels[k].sums);
			}

			radixFrames.push_back(std::move(rf));
		}
	}

	// Records the full depth-sort pipeline for this frame, leaving back-to-front draw
	// indices in outIndexBuffer[0..splatCount).
	void record(vk::raii::CommandBuffer &cmd, uint32_t frameIndex, glm::vec4 viewRow2, uint32_t splatCount, vk::raii::Buffer &outIndexBuffer) const
	{
		const RadixFrame &rf = radixFrames[frameIndex];

		// 1) Seed keyBufferA/valBufferA from splat depth (real splats) + max-key padding.
		RadixInitPushConstants initPc{.viewRow2 = viewRow2, .splatCount = splatCount, .paddedCount = radixCapacity};
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *radixInitPipeline);
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, radixInitPipelineLayout, 0, *rf.radixInitSet, nullptr);
		cmd.pushConstants<RadixInitPushConstants>(radixInitPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, initPc);
		uint32_t groups = (radixCapacity + 255u) / 256u;
		cmd.dispatch(groups, 1, 1);
		vkutil::recordComputeBarrier(cmd);

		// 2) LSD radix sort over the depth key; result ends up in keyBufferA/valBufferA.
		recordRadix(cmd, rf, kDepthKeyPasses);

		// recordRadix's internal barriers only sync compute -> compute; the final scatter's
		// write to valBufferA must also be made visible to the transfer stage before the copy below.
		const vk::MemoryBarrier2 toTransferBarrier{
		    .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
		    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
		    .dstStageMask  = vk::PipelineStageFlagBits2::eTransfer,
		    .dstAccessMask = vk::AccessFlagBits2::eTransferRead};
		cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &toTransferBarrier});

		// 3) Copy the sorted splat indices (only the first splatCount matter) into the draw-time index buffer.
		cmd.copyBuffer(*rf.valBufferA, *outIndexBuffer, vk::BufferCopy{.srcOffset = 0, .dstOffset = 0, .size = sizeof(uint32_t) * splatCount});

		const vk::MemoryBarrier2 toVertexBarrier{
		    .srcStageMask  = vk::PipelineStageFlagBits2::eTransfer,
		    .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
		    .dstStageMask  = vk::PipelineStageFlagBits2::eVertexShader,
		    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead};
		cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &toVertexBarrier});
	}

	// Headless correctness check: sort random 64-bit keys on the GPU and compare to
	// std::sort. Clobbers radixFrames[0]'s A buffers with test data; record()'s init pass
	// overwrites them unconditionally, so nothing further is needed before normal use.
	bool selfTest(const vk::raii::Device &device, const vk::raii::PhysicalDevice &physicalDevice,
	             const vk::raii::CommandPool &commandPool, const vk::raii::Queue &queue) const
	{
		const RadixFrame &rf = radixFrames[0];
		uint32_t              n  = radixCapacity;
		std::vector<uint64_t> keys(n);
		std::vector<uint32_t> vals(n);
		std::mt19937_64       rng(0x1234u);
		for (uint32_t i = 0; i < n; ++i)
		{
			keys[i] = rng();        // full 64-bit keys exercise both words (as tile keys do)
			vals[i] = i;
		}

		vkutil::uploadToDevice(device, physicalDevice, commandPool, queue, rf.keyBufferA, keys.data(), sizeof(uint64_t) * n);
		vkutil::uploadToDevice(device, physicalDevice, commandPool, queue, rf.valBufferA, vals.data(), sizeof(uint32_t) * n);

		{
			vk::CommandBufferAllocateInfo ai{.commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};
			vk::raii::CommandBuffer       cmd = std::move(device.allocateCommandBuffers(ai).front());
			cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
			recordRadix(cmd, rf, 8);        // 64-bit keys -> 8 digit passes
			cmd.end();
			queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*cmd}, nullptr);
			queue.waitIdle();
		}

		std::vector<uint64_t> outKeys(n);
		std::vector<uint32_t> outVals(n);
		vkutil::readbackFromDevice(device, physicalDevice, commandPool, queue, rf.keyBufferA, outKeys.data(), sizeof(uint64_t) * n);
		vkutil::readbackFromDevice(device, physicalDevice, commandPool, queue, rf.valBufferA, outVals.data(), sizeof(uint32_t) * n);

		bool ok = true;
		for (uint32_t i = 1; i < n && ok; ++i)
			if (outKeys[i - 1] > outKeys[i])
				ok = false;
		for (uint32_t i = 0; i < n && ok; ++i)
			if (keys[outVals[i]] != outKeys[i])
				ok = false;
		if (ok)
		{
			std::vector<uint64_t> ref = keys;
			std::sort(ref.begin(), ref.end());
			if (ref != outKeys)
				ok = false;
		}
		std::cout << "radix self-test (" << n << " keys): " << (ok ? "PASS" : "FAIL") << std::endl;
		return ok;
	}

  private:
	// One exclusive-scan level per recursion step over the block histogram.
	struct ScanLevel
	{
		vk::raii::Buffer       sums    = nullptr;
		vk::raii::DeviceMemory sumsMem = nullptr;
		uint32_t               dataLen   = 0;        // elements scanned at this level
		uint32_t               numBlocks = 0;        // ceil(dataLen / 256)
	};

	// Ping-pong A/B buffers, duplicated per frame in flight (see create()'s framesInFlight).
	struct RadixFrame
	{
		vk::raii::Buffer       keyBufferA = nullptr, keyBufferB = nullptr;
		vk::raii::DeviceMemory keyBufferAMem = nullptr, keyBufferBMem = nullptr;
		vk::raii::Buffer       valBufferA = nullptr, valBufferB = nullptr;
		vk::raii::DeviceMemory valBufferAMem = nullptr, valBufferBMem = nullptr;
		vk::raii::Buffer       blockHistBuffer    = nullptr;
		vk::raii::DeviceMemory blockHistBufferMem = nullptr;
		std::vector<ScanLevel> scanLevels;

		vk::raii::DescriptorSet              radixSetAB   = nullptr;        // A -> B
		vk::raii::DescriptorSet              radixSetBA   = nullptr;        // B -> A
		vk::raii::DescriptorSet              radixInitSet = nullptr;        // splats -> A
		std::vector<vk::raii::DescriptorSet> scanSets;                      // one per scan level
	};

	void writeRadixSet(const vk::raii::Device &device, vk::raii::DescriptorSet &set, vk::raii::Buffer &kin, vk::raii::Buffer &vin,
	                   vk::raii::Buffer &kout, vk::raii::Buffer &vout, vk::raii::Buffer &blockHist) const
	{
		vk::DescriptorBufferInfo kinInfo{.buffer = kin, .offset = 0, .range = sizeof(uint64_t) * radixCapacity};
		vk::DescriptorBufferInfo vinInfo{.buffer = vin, .offset = 0, .range = sizeof(uint32_t) * radixCapacity};
		vk::DescriptorBufferInfo koutInfo{.buffer = kout, .offset = 0, .range = sizeof(uint64_t) * radixCapacity};
		vk::DescriptorBufferInfo voutInfo{.buffer = vout, .offset = 0, .range = sizeof(uint32_t) * radixCapacity};
		vk::DescriptorBufferInfo bhInfo{.buffer = blockHist, .offset = 0, .range = sizeof(uint32_t) * radixCapacity};
		std::array w{
		    vk::WriteDescriptorSet{.dstSet = set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &kinInfo},
		    vk::WriteDescriptorSet{.dstSet = set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &vinInfo},
		    vk::WriteDescriptorSet{.dstSet = set, .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &koutInfo},
		    vk::WriteDescriptorSet{.dstSet = set, .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &voutInfo},
		    vk::WriteDescriptorSet{.dstSet = set, .dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &bhInfo}};
		device.updateDescriptorSets(w, {});
	}

	static void writeScanSet(const vk::raii::Device &device, vk::raii::DescriptorSet &set, vk::raii::Buffer &data, vk::raii::Buffer &sums)
	{
		vk::DescriptorBufferInfo dataInfo{.buffer = data, .offset = 0, .range = vk::WholeSize};
		vk::DescriptorBufferInfo sumsInfo{.buffer = sums, .offset = 0, .range = vk::WholeSize};
		std::array w{
		    vk::WriteDescriptorSet{.dstSet = set, .dstBinding = 5, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &dataInfo},
		    vk::WriteDescriptorSet{.dstSet = set, .dstBinding = 6, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &sumsInfo}};
		device.updateDescriptorSets(w, {});
	}

	// Exclusive prefix sum of blockHistBuffer, in place (multi-level; scans block sums recursively).
	void recordScan(vk::raii::CommandBuffer &cmd, const RadixFrame &rf) const
	{
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *scanBlockPipeline);
		for (size_t k = 0; k < rf.scanLevels.size(); ++k)
		{
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, scanPipelineLayout, 0, *rf.scanSets[k], nullptr);
			RadixPushConstants pc{.count = rf.scanLevels[k].dataLen, .digit = 0, .numBlocks = rf.scanLevels[k].numBlocks};
			cmd.pushConstants<RadixPushConstants>(scanPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pc);
			cmd.dispatch(rf.scanLevels[k].numBlocks, 1, 1);
			vkutil::recordComputeBarrier(cmd);
		}
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *scanAddPipeline);
		for (size_t k = rf.scanLevels.size(); k-- > 0;)
		{
			if (rf.scanLevels[k].numBlocks <= 1u)
				continue;        // deepest level was fully scanned in one block
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, scanPipelineLayout, 0, *rf.scanSets[k], nullptr);
			RadixPushConstants pc{.count = rf.scanLevels[k].dataLen, .digit = 0, .numBlocks = rf.scanLevels[k].numBlocks};
			cmd.pushConstants<RadixPushConstants>(scanPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pc);
			cmd.dispatch(rf.scanLevels[k].numBlocks, 1, 1);
			vkutil::recordComputeBarrier(cmd);
		}
	}

	// Sort rf's keyBuffer/valBuffer A ascending in place over `numPasses` 8-bit digits (result ends in A).
	void recordRadix(vk::raii::CommandBuffer &cmd, const RadixFrame &rf, uint32_t numPasses) const
	{
		for (uint32_t p = 0; p < numPasses; ++p)
		{
			const vk::raii::DescriptorSet &set = (p % 2u == 0u) ? rf.radixSetAB : rf.radixSetBA;
			RadixPushConstants             pc{.count = radixCapacity, .digit = p, .numBlocks = radixNumBlocks};

			cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *radixHistogramPipeline);
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, radixPipelineLayout, 0, *set, nullptr);
			cmd.pushConstants<RadixPushConstants>(radixPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pc);
			cmd.dispatch(radixNumBlocks, 1, 1);
			vkutil::recordComputeBarrier(cmd);

			recordScan(cmd, rf);
			vkutil::recordComputeBarrier(cmd);

			cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *radixScatterPipeline);
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, radixPipelineLayout, 0, *set, nullptr);
			cmd.pushConstants<RadixPushConstants>(radixPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pc);
			cmd.dispatch(radixNumBlocks, 1, 1);
			vkutil::recordComputeBarrier(cmd);
		}
	}

	uint32_t radixCapacity  = 0;        // padded element count (multiple of 256)
	uint32_t radixNumBlocks = 0;        // radixCapacity / 256

	// Declared before radixFrames: members destruct in reverse declaration order, and the
	// sets living in radixFrames[] must be freed back to this pool while it's still alive.
	vk::raii::DescriptorPool radixDescriptorPool = nullptr;
	std::vector<RadixFrame>  radixFrames;

	vk::raii::DescriptorSetLayout radixSetLayout         = nullptr;        // bindings 0-4
	vk::raii::DescriptorSetLayout scanSetLayout          = nullptr;        // bindings 5-6
	vk::raii::PipelineLayout      radixPipelineLayout    = nullptr;
	vk::raii::PipelineLayout      scanPipelineLayout     = nullptr;
	vk::raii::Pipeline            radixHistogramPipeline = nullptr;
	vk::raii::Pipeline            radixScatterPipeline   = nullptr;
	vk::raii::Pipeline            scanBlockPipeline      = nullptr;
	vk::raii::Pipeline            scanAddPipeline        = nullptr;

	// Depth-key init pass: splats -> (keyBufferA, valBufferA). Own set/layout (bindings 0-2).
	vk::raii::DescriptorSetLayout radixInitSetLayout      = nullptr;
	vk::raii::PipelineLayout      radixInitPipelineLayout = nullptr;
	vk::raii::Pipeline            radixInitPipeline       = nullptr;
};
