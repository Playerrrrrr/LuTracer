#pragma once
#include<cuda_AABB.hpp>
#include<vector>
#include<core_para.h>
namespace LuTracer {

	struct cuda_BVH {
		uint32_t idx = 0;//triangle_size==0ʱΪidx����ӽڵ㣬triangle_size��=0ʱidxΪ��������ʼ����
		uint32_t triangle_size = 0;
		cuda_AABB m_aabb;
	};

}