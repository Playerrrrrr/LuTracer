#pragma once
#include<host_AABB.h>
#include<vector>
#include<core_para.h>
namespace LuTracer {
	
	struct host_BVH {
		uint32_t idx = 0;//triangle_size==0ʱΪidx����ӽڵ㣬triangle_size��=0ʱidxΪ��������ʼ����
		uint32_t triangle_size = 0;
		host_AABB m_aabb;
	};
}