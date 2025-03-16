#pragma once
#include<glm\glm.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
namespace LuTracer {
	class camera;
	class ray {
		glm::vec3 origin , dir ;
		friend class camera;
	public:
		__host__ __device__ ray() {
			origin = { 1.0f, 1.0f, 1.0f };
			dir = { 3.0f, 1.0f, 1.0f };
		}
		__host__ __device__ ray(glm::vec3 ori, glm::vec3 dir) :
			origin(ori) {
			this->dir = glm::normalize(dir);
		}
		__host__ __device__ const glm::vec3& get_oirgin()const  { return origin; };
		__host__ __device__ const glm::vec3& get_dir() const { return dir; };
		__host__ __device__ const glm::vec3& get_div_dir() const { return 1.0f/ dir; };
		__host__ __device__ glm::vec3 at(float t)  const{ return origin + dir * t; }
		__host__ __device__ void set_dir(const glm::vec3& new_dir) { dir = new_dir; }
		__host__ __device__ void set_oirgin(const glm::vec3& new_oirgin) { origin = new_oirgin; }

	};
}
