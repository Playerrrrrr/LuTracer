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
		//sky color �Ժ���share
		glm::vec3* sky_color;
		glm::vec3* position_image;
		glm::vec3* normal_image;

		float* rand_map;
		unsigned int rand_map_idx;

		joint_bilateral_filter m_filter;
		int clock_rate_kHz;
		//�豸����

		unsigned int* frame_cnt;

		__device__ launch_result launch_ray(const ray& camera_ray) {
			launch_result res;

			//��Ϊcameraû��brdf������Ҫ����camera ray
			glm::vec3 Lo{ 0.0f };
			glm::vec3 history{ 1.0f };

			cuda_hit_payload hit_res;

			hit_res = BVH_test(camera_ray);

			if (!hit_res.is_hit) {//camera û���� �򷵻�sky color
				res.roughtness = -1.0f;//��pixel������պ�ʱԼ��roughness =1.0f
				res.color = *sky_color;
				return res;
			}
			//��Щ���ǵ�һ��hitʱ����Ϣ
			res.roughtness = hit_res.material->roughness;
			res.position = hit_res.position;
			res.normal = hit_res.normal;
			res.albedo = hit_res.material->base_color;

			ray t_ray = camera_ray;

			//res.dist û�о����任
			int bounce_num = 0;
			for (bounce_num = 0; bounce_num < core_para::CUDA_MAX_BOUNCE_NUM(); bounce_num++) {

				thread_bounce_num[utility_function::get_thread_idx()] = bounce_num;//�����������Ҫ��
				
				glm::vec3 emit_light = hit_res.material->emit_light * hit_res.material->emit_light_strength;//�Է���
				Lo += history * (emit_light); //�����Է���

				ray next_ray = m_sampler.importance_sampling(hit_res);//����Ҫ��hit_res��дnormal ��pos
				hit_res.L = next_ray.get_dir();//��д���䷽��
				next_ray.set_oirgin(next_ray.get_oirgin() + hit_res.geo_normal* 0.001f);//��΢ƫ�ƣ����������޹�

				{//���Բ��������Ƿ�Ϸ�
					float cosine_L_with_geo_normal = glm::dot(next_ray.get_dir(), hit_res.geo_normal);
					if (cosine_L_with_geo_normal < 0.0f) {//��������������Ƭ���棬׷��
						break;
					}
				}

				float pdf = m_sampler.pdf(hit_res);//pdf
				glm::vec3 BRDF = hit_res.material->calculate_BRDF(next_ray.get_dir(), -t_ray.get_dir(), hit_res.normal);//pdf
				float cosine_L_with_normal = glm::max(0.f, glm::dot(next_ray.get_dir(), hit_res.normal));//�����ͷ��߼н�����

				hit_res = BVH_test(next_ray);//bvh����

				if (!hit_res.is_hit) {//���δ���У����ۼ�sky light
					Lo += history * (*sky_color) * BRDF * cosine_L_with_normal / pdf;
					break;
				}

				history *= (BRDF * cosine_L_with_normal / pdf);//�ۼ�������
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
				//res��distҲҪ��������仯
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