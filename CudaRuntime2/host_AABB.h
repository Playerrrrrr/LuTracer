#pragma once
#include<glm/glm.hpp>

namespace LuTracer {

	struct host_AABB {
		glm::vec3 min{ std::numeric_limits<float>::max() }, max{ -std::numeric_limits<float>::max() };
		void insert(const glm::vec3& np) {
			min = glm::min(np, min);
			max = glm::max(max, np);
		}
		float get_area() {
			glm::vec3 len = max - min;
			return glm::dot(len, len) * 2.0f;
		}
	};
}