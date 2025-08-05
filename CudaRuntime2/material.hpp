#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<glm/glm.hpp>
#include<ray.h>
#include<core_para.h>
#include<utility.h>
namespace LuTracer {
	struct material {
		glm::vec3 base_color;
		float roughness;
		float metallic;
		float specular = 1.f;
		float specularTint = 0.0f;
		glm::vec3 emit_light{ 0.f };
		float emit_light_strength = 1.0f;

		float subsurface;
		float anisotropic;
		float sheen;
		float sheenTint;
		float clearcoat;
		float clearcoatGloss;


		//L是出射方向，V是入射方向的反方向，N是normal
		__device__ __host__ glm::vec3 calculate_BRDF(
			const glm::vec3& L,
			const glm::vec3& V,
			const glm::vec3& N)
		{
			material& para = *this;
			float NdotL = glm::dot(N, L);
			float NdotV = glm::dot(N, V);
			if (NdotL < 0 || NdotV < 0) {
				return { glm::vec3(0) };
			}

			glm::vec3 H = glm::normalize(L + V);
			float NdotH = glm::dot(N, H);
			float LdotH = glm::dot(L, H);

			// 各种颜色
			//glm::vec3 Cdlin = para.base_color;
			//float Cdlum = 0.3 * Cdlin.r + 0.6 * Cdlin.g + 0.1 * Cdlin.b;
			//glm::vec3 Ctint = (Cdlum > 0) ? (Cdlin / Cdlum) : (glm::vec3(1));
			//glm::vec3 Cspec = para.specular * glm::mix(glm::vec3(1), Ctint, para.specularTint);
			//glm::vec3 Cspec0 = glm::mix(0.08f * Cspec, Cdlin, para.metallic); // 0° 镜面反射颜色
			//float FH = utility_function::SchlickFresnel(LdotH);
			glm::vec3 F0 = glm::mix(glm::vec3{ 0.04 }, para.base_color, metallic);

			//法线分布函数
			float alpha = para.roughness * para.roughness;
			float Ds = utility_function::GTR2(NdotH, alpha);


			float Gs = utility_function::smithG_GGX(NdotL, para.roughness);
			Gs *= utility_function::smithG_GGX(NdotV, para.roughness);


			glm::vec3 diffuse = para.base_color * (1.0f-F0) * (1.0f - para.metallic) / math_const_value::PI();
			glm::vec3 specular = Gs * F0 * Ds;



			return glm::vec3{diffuse  + specular};
		}
	};
}