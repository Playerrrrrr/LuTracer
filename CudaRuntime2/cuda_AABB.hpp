#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<glm/glm.hpp>
#include<core_para.h>
#include<ray.h>
namespace LuTracer {

	struct cuda_AABB {
		glm::vec3 min, max;
		__device__ float intersect(const ray& t_ray) {
			
			glm::vec3 t_min = (min - t_ray.get_oirgin()) * (t_ray.get_div_dir());
			glm::vec3 t_max = (max - t_ray.get_oirgin()) * (t_ray.get_div_dir());
			glm::vec3 t1 = glm::min(t_min, t_max);
			glm::vec3 t2 = glm::max(t_min, t_max);
			float dst_far = glm::min(glm::min(t2.x, t2.y), t2.z);
			float dst_near = glm::max(glm::max(t1.x, t1.y), t1.z);
			bool did_hit = dst_far >= dst_near && dst_far > 0.0f;
			return did_hit ? dst_near : (core_para::FLOAT_MAX());
			
			/*
			float tmin = 0, tmax = 1e9;
			for (int a = 0; a < 3; a++)
			{
				float invD = 1.0f / t_ray.get_dir()[a];
				float t0 = (min[a] - t_ray.get_oirgin()[a]) * invD;
				float t1 = (max[a] - t_ray.get_oirgin()[a]) * invD;
				if (invD < 0.0f) {
					float temp = t0;
					t0 = t1; t1 = temp;
				}
				tmax = t1 < tmax ? t1 : tmax;   //F为两者终点的最小值
				tmin = t0 > tmin ? t0 : tmin;   //f为两者起点的最大值
				if (tmax <= tmin)       //F <= f
					return false;
			}
			return true;
			*/
		}
	};
}