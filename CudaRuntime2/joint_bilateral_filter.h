#pragma once
#include<glm/glm.hpp>
#include<utility.h>
namespace LuTracer {

	struct joint_config {
		float sigma_normal = 0.01f, 
				 sigma_position = 0.1f,
				 sigma_color = 0.05f,
				 sigma_albedo = 1.0f,
				 sigma_roughness = 1.0f;
		uint32_t kernel_size = 2;
		uint32_t buffer_width, buffer_height;
	};

	class joint_bilateral_filter {
	public:
		glm::vec3* position_buffer;
		glm::vec3* normal_buffer;
		glm::vec3* albedo_buffer;
		uint32_t* color_buffer;
		float* roughtness_buffer;//当pixel采样天空盒时约定roughness =1.0f
		uint32_t* result;
		joint_config* config;

		void init(int width,int hight, joint_config& t_config) {
			int pixel_size = width * hight;
			int pixel_bit = sizeof uint32_t;
			//check_cf(cudaMalloc((void**)&(color_buffer), pixel_size * pixel_bit));
			check_cf(cudaMalloc((void**)&(position_buffer), pixel_size * sizeof (glm::vec3)));
			check_cf(cudaMalloc((void**)&(normal_buffer), pixel_size * sizeof (glm::vec3)));
			check_cf(cudaMalloc((void**)&(albedo_buffer), pixel_size * sizeof (glm::vec3)));
			check_cf(cudaMalloc((void**)&(roughtness_buffer), pixel_size * sizeof(float)));
			check_cf(cudaMalloc((void**)&(result), pixel_size * sizeof(uint32_t)));
			check_cf(cudaMalloc((void**)&(config), sizeof(joint_config)));
			check_cf(cudaMemcpy(config, &t_config, sizeof(joint_config), cudaMemcpyHostToDevice));
		}

		__device__ void write_pixel(int idx,glm::uvec3& res) {
			unsigned char* pixel = (unsigned char*)&result[idx];
			pixel[0] = res.x;
			pixel[1] = res.y;
			pixel[2] = res.z;
			pixel[3] = 255;
		}
	};

	struct joint_bilateral_filter_displayor {
		static bool show_ui(joint_config& m_filter);
	};
}