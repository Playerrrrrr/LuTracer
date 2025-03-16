#pragma once
#include<ray.h>
#include<cuda_model.h>
#include<random>
#include<utility.h>
#include<variable.h>
namespace LuTracer {
	class uniform_sampler {
	public:
		__device__ ray static importance_sampling(const cuda_hit_payload& para) {
			//随机数生成这里有需要改进一下？
			glm::vec2 rand_num = utility_function::sobolVec2(
				pixel_sample_cnt[utility_function::get_thread_idx()] + 1 + utility_function::wang_hash(utility_function::get_thread_idx()),
				thread_bounce_num[utility_function::get_thread_idx()]
			);
			float u = math_const_value::PI() * 2.0f * rand_num.x;
			glm::vec3 dir;
			/*
			if (utility_function::get_thread_idx() == 0 && blockIdx.x==0) {
				printf("%f %f\n", rand_num.x, rand_num.y);
			}
			*/
			dir.z = rand_num.y;
			float sin_ = glm::sqrt(1.0f - dir.z * dir.z);
			dir.x = sin_ * glm::cos(u);
			dir.y = sin_ * glm::sin(u);
			ray next_ray{ para.position,
				utility_function::translate_to_normal_hemisphere(para.normal,dir)
			};
			return next_ray;
		}
		__device__ static float pdf(const cuda_hit_payload& para) {
			return 1.0f / math_const_value::PI() * 2.0f;
		}
	};

	class NDF_importance_sampler {
		uniform_sampler m_cos_sampling;
	public:

		__device__ static glm::vec3 SampleGTR2(float xi_1, float xi_2, glm::vec3 V, glm::vec3 N, float alpha) {

			float phi_h = 2.0 * math_const_value::PI() * xi_1;
			float sin_phi_h = sin(phi_h);
			float cos_phi_h = cos(phi_h);

			float cos_theta_h = sqrt((1.0 - xi_2) / (1.0 + (alpha * alpha - 1.0) * xi_2));
			float sin_theta_h = sqrt(glm::max(0.0, 1.0 - cos_theta_h * cos_theta_h));

			// 采样 "微平面" 的法向量 作为镜面反射的半角向量 h 
			glm::vec3 H = glm::vec3(sin_theta_h * cos_phi_h, sin_theta_h * sin_phi_h, cos_theta_h);
			H = utility_function::translate_to_normal_hemisphere(N,H);   // 投影到真正的法向半球

			// 根据 "微法线" 计算反射光方向
			glm::vec3 L = glm::reflect(-V, H);

			return L;
		}
		__device__ ray static importance_sampling(const cuda_hit_payload& para) {

			glm::vec2 rand_num = utility_function::sobolVec2(
				pixel_sample_cnt[utility_function::get_thread_idx()] + 1 + utility_function::rand(),
				thread_bounce_num[utility_function::get_thread_idx()]
			);

			material* t_para = para.material;
			//随机数生成这里有需要改进一下？
			float alpha_GTR2 = glm::max(0.001f, t_para->roughness * t_para->roughness);

			// 辐射度统计
			float r_diffuse = (1.0 - t_para->metallic);
			float r_specular = 1.0;
			float r_sum = r_diffuse + r_specular;

			// 根据辐射度计算概率
			float p_diffuse = r_diffuse / r_sum;
			float p_specular = r_specular / r_sum;

			float rd = utility_function::rand();

			if (rd <= p_diffuse) {
				return uniform_sampler::importance_sampling(para);
			}

			float xi_1 = rand_num.x * math_const_value::PI() * 2.0f;
			float xi_2 = rand_num.y;
			float alpha = glm::max(0.001f, t_para->roughness * t_para->roughness);
			float phi_h = 2.0 * math_const_value::PI() * xi_1;
			float sin_phi_h = sin(phi_h);
			float cos_phi_h = cos(phi_h);
			float cos_theta_h = glm::sqrt((1.0f - xi_2) / (1.0f + (alpha * alpha - 1.0f) * xi_2));
			float sin_theta_h = glm::sqrt(glm::max(0.0f, 1.0f - cos_theta_h * cos_theta_h));
			// 采样 "微平面" 的法向量 作为镜面反射的半角向量 h 
			glm::vec3 H = glm::vec3(sin_theta_h * cos_phi_h, sin_theta_h * sin_phi_h, cos_theta_h);
			H = utility_function::translate_to_normal_hemisphere(para.normal, H);   // 投影到真正的法向半球
			// 根据 "微法线" 计算反射光方向
			glm::vec3 L = glm::reflect(para.V, H);

			return { para.position ,L };

		}
		__device__ static float pdf(const cuda_hit_payload& para) {
			/*
			glm::vec3 in_dir = para.L;
			glm::vec3 out_dir = (-para.V);
			//约定参数
			glm::vec3 H = glm::normalize(in_dir + out_dir);
			float NdotH = glm::dot(para.normal, H);
			float NdotL = glm::dot(para.normal, in_dir);
			float NdotV = glm::dot(para.normal, out_dir);
			float LdotH = glm::dot(in_dir, H);
			if (NdotL < 0 || NdotV < 0) return 0;

			// 根据辐射度计算概率
			float r_diffuse = (1.0f - para.material->metallic);
			float r_specular = 1.0f;
			float r_sum = r_diffuse + r_specular;

			float p_diffuse = r_diffuse / r_sum;
			float p_specular = r_specular / r_sum;

			//漫反射
			float pdf_diffuse = uniform_sampler::pdf(para);

			//镜面反射
			float Ds = utility_function::GTR2(NdotH, glm::max(0.001f, para.material->roughness * para.material->roughness));
			float pdf_specular = Ds * NdotH / (4.0f * glm::dot(in_dir, H));

			return glm::max(1e-10f, pdf_specular * p_specular + p_diffuse * pdf_diffuse);
			*/
			glm::vec3 N = para.normal;
			glm::vec3 L = para.L;
			glm::vec3 V = (-para.V);
			float NdotL = dot(N, L);
			float NdotV = dot(N, V);
			if (NdotL < 0 || NdotV < 0) return 0;

			glm::vec3 H = normalize(L + V);
			float NdotH = dot(N, H);
			float LdotH = dot(L, H);

			// 镜面反射 -- 各向同性
			float alpha = glm::max(0.001f, para.material->roughness* para.material->roughness);
			float Ds = utility_function::GTR2(NdotH, alpha);

			// 分别计算三种 BRDF 的概率密度
			float pdf_diffuse = 1.0f / math_const_value::PI() * 2.0f;
			float pdf_specular = Ds * NdotH / (4.0 * dot(L, H));

			// 辐射度统计
			float r_diffuse = (1.0 - para.material->metallic);
			float r_specular = 1.0;
			float r_sum = r_diffuse + r_specular;

			// 根据辐射度计算选择某种采样方式的概率
			float p_diffuse = r_diffuse / r_sum;
			float p_specular = r_specular / r_sum;

			// 根据概率混合 pdf
			float pdf = p_diffuse * pdf_diffuse
				+ p_specular * pdf_specular;

			pdf = glm::max(1e-2f, pdf);
			return pdf;
		}
	};
}