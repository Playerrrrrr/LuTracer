
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<glm/glm.hpp>
#include <stdio.h>
#include<scene.h>
#include<utility.h>
#include <thrust/extrema.h>
#include<ray.h>
#include<utility.h>
using namespace LuTracer;



__device__ void write_pixel(cuda_scene& m_scene, uint32_t idx, glm::vec3 radiance) {
	unsigned int pixel_frame_cnt = *m_scene.frame_cnt;
	unsigned char* pixel_color = (unsigned char*)&m_scene.m_pixel[idx];
	glm::vec3& pixel_radiance = m_scene.m_pixel_radiance[idx];
	pixel_radiance += radiance;
	glm::vec3 mean_of_radiance = pixel_radiance / (float)pixel_frame_cnt;
	

	glm::vec3 pixel_radiance_mapping = utility_function::tone_mapping(//这个limit是不是蓝色bug来源？还真是
		mean_of_radiance, .5f
	);

	pixel_color[0] = pixel_radiance_mapping.x * 255;
	pixel_color[1] = pixel_radiance_mapping.y * 255;
	pixel_color[2] = pixel_radiance_mapping.z * 255;
	pixel_color[3] = 255;

}

//__device__ sphere s1{ glm::vec3(0,0,0),0.5 }; 错误s1的初始化调用了初始化函数


#define D1_D2
//#define D2_D1

#ifdef D2_D1
//二维网格+一维线程块
__device__ int get_idx() {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = blockIdx.y;
	int nx = gridDim.x * blockDim.x;
	return iy * nx + ix;
}
#endif // D2_D1


#ifdef D1_D2
//一维网格+二维线程块

#endif // D1_D2



//为什么cuda_scene&就报错？可能是在传进来之前是host mem，不能直接引用到device mem
__global__ void launch_ray(cuda_scene scene) {
	rand_map = scene.rand_map;
	//计算对应像素的小标
	float task_num_per_thread_float = (float)scene.image_hight * scene.image_width / (blockDim.x * gridDim.x);
	float begin_idx_float = task_num_per_thread_float * utility_function::get_global_idx();
	float next_begin_idx_float = task_num_per_thread_float * (blockIdx.x * blockDim.x + threadIdx.x + 1);
	int begin_idx = glm::ceil(begin_idx_float);
	int next_idx = glm::ceil(next_begin_idx_float);
	int task_num_per_thread = next_idx - begin_idx;
	//计算下一个线程的起始小标

	for (int i = 0; i < task_num_per_thread; i++) {
		glm::vec2 pixel_pos{ (begin_idx + i) % scene.image_width,(begin_idx + i) / scene.image_hight };
		//glm::vec2 pixel_pos = scene.pixel_pos[begin_idx + i];
		//int pixel_pos_one_d = pixel_pos.y * scene.image_width + pixel_pos.x;
		//生成随机数需要的，这里要改
		ray& t_ray = scene.camera_ray[begin_idx + i];

		auto launch_res = scene.launch_ray(t_ray);

		//显示时间
		//write_pixel(scene, begin_idx + i, glm::vec3{ glm::max(time_recoder,0.f)});

		scene.m_filter.albedo_buffer[begin_idx+i] = launch_res.albedo;
		scene.m_filter.normal_buffer[begin_idx + i] = launch_res.normal;
		scene.m_filter.roughtness_buffer[begin_idx + i] = launch_res.roughtness;
		scene.m_filter.position_buffer[begin_idx + i] = launch_res.position;

		//write_pixel(scene, pixel_pos_one_d, glm::vec3{1.0f});
		write_pixel(scene, begin_idx + i, launch_res.color);
	}
}



cudaError cuda_render(cuda_scene scene) {
	dim3 grid_dim{ core_para::GRID_DIM_X(),core_para::GRID_DIM_Y() };
	dim3 block_dim{ core_para::BLOCK_DIM_X(),core_para::BLOCK_DIM_Y() };
	launch_ray << <grid_dim, block_dim >> > (scene);
	cudaDeviceSynchronize();
	return cudaSuccess;
}