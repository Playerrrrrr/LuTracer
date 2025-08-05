#pragma once

namespace LuTracer {
	//记得改cuda那边
	struct core_para {
		__host__ __device__ static constexpr const unsigned int BVH_MAX_FACE() { return 16; };
		__host__ __device__ static constexpr const unsigned int BVH_MAX_DEEPTH() { return 32; };
		__host__ __device__ static constexpr const float FLOAT_MAX() { return 1e10; };
		__host__ __device__ static constexpr const unsigned int CUDA_MAX_MODEL_NUM() { return 8; };
		__host__ __device__ static constexpr const unsigned int CUDA_MAX_BOUNCE_NUM() { return 4; };
		__host__ __device__ static constexpr const unsigned int IMAGE_WIDTH() { return 1920; };
		__host__ __device__ static constexpr const unsigned int IMAGE_HEIGHT() { return 1080; };
		__host__ __device__ static constexpr const unsigned int GRID_DIM_X() { return 16384; };
		__host__ __device__ static constexpr const unsigned int GRID_DIM_Y() { return 1; };
		__host__ __device__ static constexpr const unsigned int GRID_DIM_Z() { return 1; };
		__host__ __device__ static constexpr const unsigned int BLOCK_DIM_X() { return 128; };
		__host__ __device__ static constexpr const unsigned int BLOCK_DIM_Y() { return 1; };
		__host__ __device__ static constexpr const unsigned int BLOCK_DIM_Z() { return 1; };
		__host__ __device__ static constexpr const unsigned int RAND_MAP_SIZE() { return 1<<16; };

	};

	struct math_const_value {
		__host__ __device__ static constexpr const float PI() { return 3.1415926f; };
		__host__ __device__ static constexpr const float PI_V() { return 3.1415926f; };
	};

}