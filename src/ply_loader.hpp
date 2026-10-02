#pragma once

#include <glm/glm.hpp>
#include <string>
#include <vector>

// Count of higher-order SH coefficients per color channel for degree <= 3: bands 1-3
// contribute 3 + 5 + 7 = 15 (band 0, the diffuse term, lives in GpuSplat::color instead).
constexpr int kShRestCoeffs = 15;

// GPU-side per-splat record. Laid out to match the std430 `Splat` struct in shader.slang,
// compute.slang and radix.slang (the latter two only ever read `position`, but must still
// agree on the struct's total size so indexed access into the shared splat buffer lines up).
// All heavy per-splat math (covariance, activations) is done once here at load time; SH
// evaluation is view-dependent and happens per-frame in the vertex shader instead.
struct GpuSplat
{
	glm::vec3 position;                     // world-space mean
	float     opacity;                      // sigmoid(opacity) from the ply
	glm::vec4 color;                        // rgb = SH_C0 * f_dc (band 0, undilated), a = unused
	glm::vec4 cov_a;                        // Sigma[0][0], Sigma[0][1], Sigma[0][2], Sigma[1][1]
	glm::vec4 cov_b;                        // Sigma[1][2], Sigma[2][2], pad, pad
	glm::vec4 shRest[kShRestCoeffs];        // bands 1-3, raw ply coefficients; rgb used, a pad
};
static_assert(sizeof(GpuSplat) == 64 + 16 * kShRestCoeffs, "GpuSplat layout drifted from std430 Splat");

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
// f_rest_0.. (higher-order SH) is optional; missing coefficients default to 0, which
// reproduces the old flat SH-degree-0 look.
Scene loadPly(const std::string &path);
