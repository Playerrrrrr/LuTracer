#pragma once
#include<ray.h>
#include<cuda_model.h>
#include<core_para.h>
#include<sampler.hpp>
#include<joint_bilateral_filter.h>

namespace LuTracer {

	struct launch_result {
		float roughtness;
		glm::vec3 position, normal, albedo;
		glm::vec3 color;
	};

	struct cuda_scene {
		size_t image_width, image_hight;

		ray* camera_ray;
		uint32_t* m_pixel;
		glm::vec3* m_pixel_radiance;

		cuda_model m_models[8];
		uint32_t model_size;

		NDF_importance_sampler m_sampler;
		//sky color 以后变成share
		glm::vec3* sky_color;
		glm::vec3* position_image;
		glm::vec3* normal_image;

		float* rand_map;
		unsigned int rand_map_idx;

		joint_bilateral_filter m_filter;
		int clock_rate_kHz;
		//设备数据

		unsigned int* frame_cnt;

		__device__ launch_result launch_ray(const ray& camera_ray) {
			launch_result res;

			//因为camera没有brdf，所以要特判camera ray
			glm::vec3 Lo{ 0.0f };
			glm::vec3 history{ 1.0f };

			cuda_hit_payload hit_res;

			hit_res = BVH_test(camera_ray);

			if (!hit_res.is_hit) {//camera 没击中 则返回sky color
				res.roughtness = -1.0f;//当pixel采样天空盒时约定roughness =1.0f
				res.color = *sky_color;
				return res;
			}
			//这些都是第一次hit时的信息
			res.roughtness = hit_res.material->roughness;
			res.position = hit_res.position;
			res.normal = hit_res.normal;
			res.albedo = hit_res.material->base_color;

			ray t_ray = camera_ray;

			//res.dist 没有经过变换
			int bounce_num = 0;
			for (bounce_num = 0; bounce_num < core_para::CUDA_MAX_BOUNCE_NUM(); bounce_num++) {

				thread_bounce_num[utility_function::get_thread_idx()] = bounce_num;//生成随机树需要的
				
				glm::vec3 emit_light = hit_res.material->emit_light * hit_res.material->emit_light_strength;//自发光
				Lo += history * (emit_light); //计算自发光

				ray next_ray = m_sampler.importance_sampling(hit_res);//这里要求hit_res填写normal ，pos
				hit_res.L = next_ray.get_dir();//填写出射方向
				next_ray.set_oirgin(next_ray.get_oirgin() + hit_res.geo_normal* 0.001f);//稍微偏移，以免乱伤无辜

				{//测试采样方向是否合法
					float cosine_L_with_geo_normal = glm::dot(next_ray.get_dir(), hit_res.geo_normal);
					if (cosine_L_with_geo_normal < 0.0f) {//采样到三角形面片下面，追踪
						break;
					}
				}

				float pdf = m_sampler.pdf(hit_res);//pdf
				glm::vec3 BRDF = hit_res.material->calculate_BRDF(next_ray.get_dir(), -t_ray.get_dir(), hit_res.normal);//pdf
				float cosine_L_with_normal = glm::max(0.f, glm::dot(next_ray.get_dir(), hit_res.normal));//出射光和法线夹角余弦

				hit_res = BVH_test(next_ray);//bvh测试

				if (!hit_res.is_hit) {//如果未命中，则累计sky light
					Lo += history * (*sky_color) * BRDF * cosine_L_with_normal / pdf;
					break;
				}

				history *= (BRDF * cosine_L_with_normal / pdf);//累计吞吐量
				t_ray = next_ray;
			}
			res.color = Lo;

			return res;

			// history  cosine_L_with_normal emit_light pdf cosine_L_with_geo_normal
			//position
		}

		__device__ cuda_hit_payload BVH_test(const ray& t_ray) {
			cuda_hit_payload hit_res;
			hit_res.is_hit = false;
			for (int model_idx = 0; model_idx < model_size; model_idx++) {

				glm::vec4 trans_ray_pos = *m_models[model_idx].m_translate_matrix * glm::vec4{ t_ray.get_oirgin(),1.0f };
				glm::vec4 trans_ray_dir = *m_models[model_idx].m_translate_matrix * glm::vec4{ t_ray.get_dir() + t_ray.get_oirgin(),1.0f };
				ray trans_ray{ trans_ray_pos,glm::normalize(trans_ray_dir - trans_ray_pos) };

				cuda_hit_payload res = m_models[model_idx].hit_test(trans_ray);
				//res的dist也要经过矩阵变化
				float dist = glm::length(res.position - t_ray.get_oirgin());
				if (res.is_hit && hit_res.dist > dist) {
					hit_res = res;
					hit_res.dist = dist;
					hit_res.V = glm::normalize(res.position - t_ray.get_oirgin());
				}
			}
			return hit_res;
		}
	};
}