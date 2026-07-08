#pragma once

#include <glm/glm.hpp>
#include <string>
#include <vector>

// GPU-side per-splat record. Laid out to match the std430 `Splat` struct in the
// Slang shader: 64 bytes, 16-byte aligned. All heavy per-splat math (covariance,
// SH degree-0 color, activation functions) is done once here at load time.
struct GpuSplat
{
	glm::vec3 position;        // world-space mean
	float     opacity;         // sigmoid(opacity) from the ply
	glm::vec4 color;           // rgb = SH degree-0 color, a = unused
	glm::vec4 cov_a;           // Sigma[0][0], Sigma[0][1], Sigma[0][2], Sigma[1][1]
	glm::vec4 cov_b;           // Sigma[1][2], Sigma[2][2], pad, pad
};
static_assert(sizeof(GpuSplat) == 64, "GpuSplat must stay 64 bytes for std430");

struct Scene
{
	std::vector<GpuSplat> splats;
	glm::vec3             aabbMin{0.0f};
	glm::vec3             aabbMax{0.0f};

	glm::vec3 center() const { return 0.5f * (aabbMin + aabbMax); }
	float     radius() const { return 0.5f * glm::length(aabbMax - aabbMin); }
};

// Load a pretrained 3D Gaussian Splatting .ply (binary_little_endian, float32
// properties). Throws std::runtime_error on any format it does not understand.
Scene loadPly(const std::string &path);
