#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<core_para.h>
namespace LuTracer {

	__shared__ unsigned int pix_x[
		core_para::BLOCK_DIM_X() * core_para::BLOCK_DIM_Y() * core_para::BLOCK_DIM_Z()

	];
	__shared__ unsigned int pix_y[
		core_para::BLOCK_DIM_X() * core_para::BLOCK_DIM_Y() * core_para::BLOCK_DIM_Z()

	];
	__shared__ unsigned int thread_bounce_num[
		core_para::BLOCK_DIM_X() * core_para::BLOCK_DIM_Y() * core_para::BLOCK_DIM_Z()

	];

	__shared__ unsigned int pixel_sample_cnt[
		core_para::BLOCK_DIM_X() * core_para::BLOCK_DIM_Y() * core_para::BLOCK_DIM_Z()
	];


	__device__ float* rand_map;//全局可访问的scene
	__device__ unsigned int rand_map_idx = 0;//全局可访问的scene

}