#include "ply_loader.hpp"

#include <glm/gtc/quaternion.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace
{
	constexpr float SH_C0 = 0.28209479177387814f;        // 0-th band spherical harmonic

	float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

	size_t typeSize(const std::string &type)
	{
		if (type == "char" || type == "int8" || type == "uchar" || type == "uint8")
			return 1;
		if (type == "short" || type == "int16" || type == "ushort" || type == "uint16")
			return 2;
		if (type == "int" || type == "int32" || type == "uint" || type == "uint32" || type == "float" || type == "float32")
			return 4;
		if (type == "double" || type == "float64")
			return 8;
		return 0;
	}

	struct Property
	{
		std::string type;
		size_t      size   = 0;
		size_t      offset = 0;        // byte offset within a record
	};

	struct Element
	{
		std::string                               name;
		size_t                                    count  = 0;
		size_t                                    stride = 0;
		std::vector<uint8_t>                      data;        // count * stride bytes
		std::unordered_map<std::string, Property> byName;

		const uint8_t *record(size_t i) const { return data.data() + i * stride; }
		bool           has(const char *n) const { return byName.count(n) != 0; }
	};

	float readFloat(const uint8_t *rec, const Property &p)
	{
		const uint8_t *ptr = rec + p.offset;
		if (p.type == "float" || p.type == "float32")
		{
			float v;
			std::memcpy(&v, ptr, sizeof(v));
			return v;
		}
		if (p.type == "double" || p.type == "float64")
		{
			double v;
			std::memcpy(&v, ptr, sizeof(v));
			return static_cast<float>(v);
		}
		throw std::runtime_error("ply property '" + p.type + "' is not a float type");
	}

	// Build a GpuSplat from activated inputs: world position, world-space scale (already
	// exp'd), a rotation quaternion, final linear RGB, and final opacity (0..1).
	GpuSplat buildSplat(const glm::vec3 &pos, const glm::vec3 &scale, glm::quat q,
	                    const glm::vec3 &rgb, float opacity)
	{
		q = glm::normalize(q);
		glm::mat3 R     = glm::mat3_cast(q);
		glm::mat3 M     = R * glm::mat3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
		glm::mat3 sigma = M * glm::transpose(M);

		GpuSplat s;
		s.position = pos;
		s.opacity  = opacity;
		s.color    = glm::vec4(rgb, 1.0f);
		s.cov_a    = glm::vec4(sigma[0][0], sigma[0][1], sigma[0][2], sigma[1][1]);
		s.cov_b    = glm::vec4(sigma[1][2], sigma[2][2], 0.0f, 0.0f);
		return s;
	}

	// --- Uncompressed 3DGS: one float32 vertex record per Gaussian ---
	void decodeUncompressed(const Element &v, std::vector<GpuSplat> &out)
	{
		auto req = [&](const char *n) -> const Property & {
			auto it = v.byName.find(n);
			if (it == v.byName.end())
				throw std::runtime_error(std::string("ply missing required property: ") + n);
			return it->second;
		};
		const Property &px = req("x"), &py = req("y"), &pz = req("z");
		const Property &c0 = req("f_dc_0"), &c1 = req("f_dc_1"), &c2 = req("f_dc_2");
		const Property &op = req("opacity");
		const Property &s0 = req("scale_0"), &s1 = req("scale_1"), &s2 = req("scale_2");
		const Property &r0 = req("rot_0"), &r1 = req("rot_1"), &r2 = req("rot_2"), &r3 = req("rot_3");

		out.reserve(v.count);
		for (size_t i = 0; i < v.count; ++i)
		{
			const uint8_t *rec = v.record(i);
			glm::vec3      pos{readFloat(rec, px), readFloat(rec, py), readFloat(rec, pz)};
			glm::vec3      scale{std::exp(readFloat(rec, s0)), std::exp(readFloat(rec, s1)), std::exp(readFloat(rec, s2))};
			glm::quat      q{readFloat(rec, r0), readFloat(rec, r1), readFloat(rec, r2), readFloat(rec, r3)};        // (w,x,y,z)
			glm::vec3      rgb{0.5f + SH_C0 * readFloat(rec, c0), 0.5f + SH_C0 * readFloat(rec, c1), 0.5f + SH_C0 * readFloat(rec, c2)};
			out.push_back(buildSplat(pos, scale, q, rgb, sigmoid(readFloat(rec, op))));
		}
	}
}        // namespace

Scene loadPly(const std::string &path)
{
	std::ifstream file(path, std::ios::binary);
	if (!file.is_open())
		throw std::runtime_error("failed to open ply file: " + path);

	std::string line;
	if (!std::getline(file, line) || line.rfind("ply", 0) != 0)
		throw std::runtime_error("not a ply file: " + path);

	// --- Parse the ASCII header into a list of elements, each with ordered properties ---
	bool                 littleEndian = false;
	std::vector<Element> elements;
	while (std::getline(file, line))
	{
		if (!line.empty() && line.back() == '\r')
			line.pop_back();
		std::istringstream iss(line);
		std::string        tok;
		iss >> tok;

		if (tok == "format")
		{
			std::string fmt;
			iss >> fmt;
			if (fmt == "binary_little_endian")
				littleEndian = true;
			else
				throw std::runtime_error("unsupported ply format '" + fmt + "' (need binary_little_endian)");
		}
		else if (tok == "element")
		{
			Element e;
			iss >> e.name >> e.count;
			elements.push_back(std::move(e));
		}
		else if (tok == "property")
		{
			if (elements.empty())
				throw std::runtime_error("ply property outside of an element");
			Element    &e = elements.back();
			Property    p;
			std::string name;
			iss >> p.type;
			// Lists are not used by the formats we support.
			if (p.type == "list")
				throw std::runtime_error("ply list properties are not supported");
			iss >> name;
			p.size = typeSize(p.type);
			if (p.size == 0)
				throw std::runtime_error("unknown ply property type: " + p.type);
			p.offset = e.stride;
			e.stride += p.size;
			e.byName[name] = p;
		}
		else if (tok == "end_header")
		{
			break;
		}
	}
	if (!littleEndian)
		throw std::runtime_error("ply header missing binary_little_endian format");

	// --- Read each element's binary block, in header order ---
	for (Element &e : elements)
	{
		e.data.resize(e.count * e.stride);
		file.read(reinterpret_cast<char *>(e.data.data()), static_cast<std::streamsize>(e.data.size()));
		if (static_cast<size_t>(file.gcount()) != e.data.size())
			throw std::runtime_error("ply body truncated reading element '" + e.name + "'");
	}

	auto find = [&](const char *name) -> const Element * {
		for (const Element &e : elements)
			if (e.name == name)
				return &e;
		return nullptr;
	};

	const Element *vertex = find("vertex");
	if (!vertex || vertex->count == 0)
		throw std::runtime_error("ply has no vertex element");

	Scene scene;
	decodeUncompressed(*vertex, scene.splats);

	// Bounds for the initial camera placement.
	glm::vec3 lo(std::numeric_limits<float>::max());
	glm::vec3 hi(std::numeric_limits<float>::lowest());
	for (const GpuSplat &s : scene.splats)
	{
		lo = glm::min(lo, s.position);
		hi = glm::max(hi, s.position);
	}
	scene.aabbMin = lo;
	scene.aabbMax = hi;
	return scene;
}
