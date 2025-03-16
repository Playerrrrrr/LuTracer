#include "model.h"
#include<imgui.h>
#include<string>
#include<iostream>
namespace LuTracer {
	__host__ __device__ void sphere::set_position(const glm::vec3& pos)
	{
		this->pos = pos;
	}
	__host__ __device__ void sphere::offset_position(const glm::vec3 pos_offest)
	{
		this->pos += pos_offest;
	}
	__host__ __device__ void sphere::rotate()
	{
	}

}