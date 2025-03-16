#pragma once
#include<host_AABB.h>
#include<vector>
#include<core_para.h>
namespace LuTracer {
	
	struct host_BVH {
		uint32_t idx = 0;//triangle_size==0时为idx编号子节点，triangle_size！=0时idx为三角形起始坐标
		uint32_t triangle_size = 0;
		host_AABB m_aabb;
	};
}