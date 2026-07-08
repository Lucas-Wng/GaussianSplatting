#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>

// Simple free-fly / FPS camera: WASD-style translation plus mouse-look.
// World up is +Y. yaw/pitch are in radians.
class Camera
{
  public:
	enum class Move
	{
		Forward,
		Backward,
		Left,
		Right,
		Up,
		Down
	};

	glm::vec3 position{0.0f, 0.0f, 3.0f};
	float     yaw   = -glm::half_pi<float>();        // look toward -Z by default
	float     pitch = 0.0f;
	float     moveSpeed = 2.0f;                      // world units / second
	float     mouseSensitivity = 0.0025f;            // radians / pixel

	glm::vec3 front() const
	{
		return glm::normalize(glm::vec3{
		    std::cos(pitch) * std::cos(yaw),
		    std::sin(pitch),
		    std::cos(pitch) * std::sin(yaw)});
	}

	glm::mat4 viewMatrix() const
	{
		return glm::lookAt(position, position + front(), worldUp);
	}

	void processKeyboard(Move dir, float dt)
	{
		const glm::vec3 f = front();
		const glm::vec3 right = glm::normalize(glm::cross(f, worldUp));
		const float     v = moveSpeed * dt;
		switch (dir)
		{
			case Move::Forward:  position += f * v; break;
			case Move::Backward: position -= f * v; break;
			case Move::Left:     position -= right * v; break;
			case Move::Right:    position += right * v; break;
			case Move::Up:       position += worldUp * v; break;
			case Move::Down:     position -= worldUp * v; break;
		}
	}

	// dx/dy are raw cursor deltas in pixels (dy positive = cursor moved down).
	void processMouse(float dx, float dy)
	{
		yaw += dx * mouseSensitivity;
		pitch -= dy * mouseSensitivity;
		const float limit = glm::radians(89.0f);
		pitch = std::clamp(pitch, -limit, limit);
	}

  private:
	static constexpr glm::vec3 worldUp{0.0f, 1.0f, 0.0f};
};
