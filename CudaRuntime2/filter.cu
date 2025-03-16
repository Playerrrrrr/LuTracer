#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<glm/glm.hpp>
#include<core_para.h>
#include<joint_bilateral_filter.h>
#include<utility.h>
namespace LuTracer {


	__device__ float filtering(glm::ivec2 center, glm::ivec2 edge, joint_bilateral_filter& m_filter) {
		int idx_of_edge = m_filter.config->buffer_width * edge.y + edge.x;
		int idx_of_center = m_filter.config->buffer_width * center.y + center.x;

		glm::vec3& normal_e = m_filter.normal_buffer[idx_of_edge];
		glm::vec3& position_e = m_filter.position_buffer[idx_of_edge];
		glm::vec3& albedo_e = m_filter.albedo_buffer[idx_of_edge];
		glm::vec3 color_e = glm::vec3{
			((unsigned char*)&m_filter.color_buffer[idx_of_edge])[0],
			((unsigned char*)&m_filter.color_buffer[idx_of_edge])[1],
			((unsigned char*)&m_filter.color_buffer[idx_of_edge])[2]
		} / 255.f;
		float roughness_e = m_filter.roughtness_buffer[idx_of_edge];

		glm::vec3& normal_c = m_filter.normal_buffer[idx_of_center];
		glm::vec3& position_c = m_filter.position_buffer[idx_of_center];
		glm::vec3& albedo_c = m_filter.albedo_buffer[idx_of_center];
		glm::vec3 color_c = glm::vec3{
			((unsigned char*)&m_filter.color_buffer[idx_of_center])[0],
			((unsigned char*)&m_filter.color_buffer[idx_of_center])[1],
			((unsigned char*)&m_filter.color_buffer[idx_of_center])[2]
		} / 255.f;
		float roughness_c = m_filter.roughtness_buffer[idx_of_center];

		//position
		float len = glm::length(position_c - position_e);
		float position_w = len * len;
		position_w /= m_filter.config->sigma_position;
		//normal
		float dot_normal = glm::dot(normal_e, normal_c);
		float normal_w = 1.0f - dot_normal * dot_normal;
		normal_w /= m_filter.config->sigma_normal;
		//albedo
		float delta_len_albedo = glm::length(albedo_c - albedo_e);
		float albedo_w = delta_len_albedo * delta_len_albedo;
		albedo_w /= m_filter.config->sigma_albedo;
		//roughness
		float roughness_w = (roughness_e - roughness_c) * (roughness_e - roughness_c);
		roughness_w /= m_filter.config->sigma_roughness;
		//color
		float delta_len_color = glm::length(color_c - color_e);
		float color_w = delta_len_color * delta_len_color;
		color_w /= m_filter.config->sigma_color;

		float weight = glm::exp(
			-position_w - normal_w - albedo_w - roughness_w - color_w
		);
		return weight;
	}

	__global__ void cuda_filter(joint_bilateral_filter m_filter) {
		float task_num_per_thread_float = (float)m_filter.config->buffer_width * m_filter.config->buffer_height / (blockDim.x * gridDim.x);
		float begin_idx_float = task_num_per_thread_float * utility_function::get_global_idx();
		float next_begin_idx_float = task_num_per_thread_float * (blockIdx.x * blockDim.x + threadIdx.x + 1);
		int begin_idx = glm::ceil(begin_idx_float);
		int next_idx = glm::ceil(next_begin_idx_float);
		int task_num_per_thread = next_idx - begin_idx;
		//计算下一个线程的起始小标

		for (int i = 0; i < task_num_per_thread; i++) {
			glm::vec2 center{ (begin_idx + i) % core_para::IMAGE_WIDTH() ,
							  (begin_idx + i) / core_para::IMAGE_WIDTH() };
			int kernel_size = m_filter.config->kernel_size;
			if (m_filter.roughtness_buffer[begin_idx + i] < 0.0f) {//命中天空盒
				m_filter.result[begin_idx + i] = m_filter.color_buffer[begin_idx + i];
				continue;
			}
			glm::vec3 sum_of_weighted_values{ 0.0f }, sum_of_wieght{ 0.0f };
			for (int edge_x_offest = -kernel_size; edge_x_offest <= kernel_size; edge_x_offest++) {
				for (int edge_y_offest = -kernel_size; edge_y_offest <= kernel_size; edge_y_offest++) {
					glm::ivec2 edge{ center.x + edge_x_offest,center.y + edge_y_offest };
					int edge_idx = edge.y * m_filter.config->buffer_width + edge.x;
					//是否为有效核
					if (edge.x < 0 || edge.x >= m_filter.config->buffer_width
						|| edge.y < 0 || edge.y >= m_filter.config->buffer_height)
						continue;
					if (m_filter.roughtness_buffer[edge_idx] < 0.0f) {//天空盒像素不被考虑
						continue;
					}
					unsigned char* color = (unsigned char*)&m_filter.color_buffer[edge_idx];
					glm::vec3 color_glm{ (int)color[0],(int)color[1] ,(int)color[2] };
					float weight = filtering(center, edge, m_filter);
					sum_of_weighted_values += weight * color_glm;
					sum_of_wieght += weight;
				}
			}
			glm::uvec3 result = sum_of_weighted_values / sum_of_wieght;
			m_filter.write_pixel(begin_idx + i, result);
		}
	}


}

cudaError joint_bilateral_filtering(LuTracer::joint_bilateral_filter m_filter) {
	dim3 grid_dim{ LuTracer::core_para::GRID_DIM_X(),LuTracer::core_para::GRID_DIM_Y() };
	dim3 block_dim{ LuTracer::core_para::BLOCK_DIM_X(),LuTracer::core_para::BLOCK_DIM_Y() };
	LuTracer::cuda_filter << <grid_dim, block_dim >> > (m_filter);
	cudaDeviceSynchronize();
	return cudaSuccess;
}