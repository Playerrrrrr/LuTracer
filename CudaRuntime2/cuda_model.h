#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<glm/glm.hpp>
#include<ray.h>
#include<cuda_AABB.hpp>
#include<cuda_BVH.hpp>
#include<material.hpp>

namespace LuTracer {

	struct intersection_result {
		bool hit;          // 是否命中
		float t;           // 光线参数 (沿 dir 的距离)
		float u, v;        // 重心坐标 (u, v)，满足 u >= 0, v >= 0, u + v <= 1
	};

	struct cuda_vertex {
		glm::vec3 pos, normal;
		glm::vec2 uv;
		glm::vec3 tangent, bitangent;
	};

	struct cuda_mesh_face {
		uint32_t idx[3];
	};

	//每个mesh都有一个AABB
	struct cuda_mesh {
		cuda_vertex* m_vertex = nullptr;
		cuda_mesh_face* m_face = nullptr;
		uint32_t vertex_size = 0, m_face_size = 0;

		cuda_BVH* m_BVH;
		uint32_t BVH_size;
	};



	struct cuda_hit_payload {
		bool is_hit = false;

		glm::vec3 position;//击中点的空间位置
		float dist = core_para::FLOAT_MAX();//击中点离hit ray原点的距离

		glm::vec2 uv;//击中三角形的uv
		glm::vec3 geo_normal;//三角形的法线
		glm::vec3 normal;//采样得到的法线

		glm::vec3 V, L;//V是入射方向，L是出射方向

		material* material = nullptr;//击中点的material

		int intersect_cnt = 0;
	};

	struct cuda_model {
		cuda_mesh m_meshs[5];
		uint32_t mesh_size = 0;

		cuda_AABB* m_ab_box = nullptr;

		glm::mat4x4* m_translate_matrix = nullptr;
		glm::mat4x4* m_inv_translate_matrix = nullptr;
		material* m_material;

		__host__ void set_translate_matrix(glm::mat4x4 mat) {
			
		}

		__device__ intersection_result ray_triangle_intersect(
			const glm::vec3& orig,
			const glm::vec3& dir,
			const glm::vec3& v0, 
			const glm::vec3& v1, 
			const glm::vec3& v2
		)
		{
			intersection_result res;
			res.hit = false;
			glm::vec3 E1 = v1 - v0;
			glm::vec3 E2 = v2 - v0;
			glm::vec3 S = orig - v0;
			glm::vec3 S1 = glm::cross(dir, E2);
			glm::vec3 S2 = glm::cross(S, E1);
			float coeff = 1.0f / glm::dot(S1, E1); // 共同系数
			float t = coeff * glm::dot(S2, E2);
			float b1 = coeff * glm::dot(S1, S);
			float b2 = coeff * glm::dot(S2, dir);
			if (t >= 0 && b1 >= 0 && b2 >= 0 && (1 - b1 - b2) >= 0) {
				res.hit = true;
				res.t = t;
				res.u = b1;
				res.v = b2;
			}
			return res;
		}


		//幽灵：bug，两个一样的函数一个加了#ifdefine后就报错了

		__device__ cuda_hit_payload hit_test(const ray& t_ray) {
			//暴力版本
			/*
			cuda_hit_payload res;
			res.dist = 1e9;
			res.is_hit = false;

			if (!m_ab_box.intersect(t_ray))
				return res;

			for (int i = 0; i < mesh_size; i++) {
				cuda_mesh& t_mesh = m_meshs[i];
				cuda_vertex* vertexes = t_mesh.m_vertex;
				for (int j = 0; j < t_mesh.m_face_size; j++) {
					cuda_mesh_face& t_face = t_mesh.m_face[j];
					intersection_result t_res = ray_triangle_intersect(
						t_ray.get_oirgin(),
						t_ray.get_dir(),
						vertexes[t_face.idx[0]].pos,
						vertexes[t_face.idx[1]].pos,
						vertexes[t_face.idx[2]].pos
					);
					if (t_res.hit && t_res.t < res.dist) {
						res.dist = t_res.t;
						res.normal =  vertexes[t_face.idx[0]].normal * (1.0f - t_res.u - t_res.v)
										+ vertexes[t_face.idx[1]].normal * (t_res.u)
										+ vertexes[t_face.idx[2]].normal * (t_res.v);
						res.is_hit = true;
						res.position = t_ray.at(t_res.t);
					}
				}
			}
			return res;
			*/
			//bvh版本
			cuda_hit_payload res;
			res.dist = core_para::FLOAT_MAX();
			res.is_hit = false;
			if (!(*m_ab_box).intersect(t_ray))
				return res;
			//BVH示例作为node存储版本
			/*
				
						cuda_BVH nodes[core_para::BVH_MAX_DEEPTH() + 1];
			

			for (int i = 0; i < mesh_size; i++) {
				cuda_mesh& t_mesh = m_meshs[i];

				int stack_idx = 0;
				nodes[stack_idx++] = t_mesh.m_BVH[0];//加入根节点,可以用下标代替，以减少栈空间

				while (stack_idx > 0) {

					cuda_BVH node = nodes[--stack_idx];

					if (node.m_aabb.intersect(t_ray)) {
						if (node.left_child_idx == -1) //子节点，BVH只有有两个节点的情况
						{
							for (int tri_idx_offset = 0; tri_idx_offset < node.triangle_size; tri_idx_offset++) 
							{//检测三角形
								cuda_mesh_face& mesh_face = t_mesh.m_face[node.begin_idx + tri_idx_offset];
								cuda_vertex* vertexes = t_mesh.m_vertex;
								intersection_result t_res =  ray_triangle_intersect(
									t_ray.get_oirgin(), 
									t_ray.get_dir(),
									vertexes[mesh_face.idx[0]].pos,
									vertexes[mesh_face.idx[1]].pos,
									vertexes[mesh_face.idx[2]].pos
								);
								if (t_res.hit && t_res.t < res.dist) {
									res.dist = t_res.t;
									res.normal = vertexes[mesh_face.idx[0]].normal * (1.0f - t_res.u - t_res.v)
										+ vertexes[mesh_face.idx[1]].normal * (t_res.u)
										+ vertexes[mesh_face.idx[2]].normal * (t_res.v);
									res.is_hit = true;
									res.position = t_ray.at(t_res.t);
								}
							}
						}
						else {
							nodes[stack_idx++] = t_mesh.m_BVH[node.left_child_idx];
							nodes[stack_idx++] = t_mesh.m_BVH[node.right_child_idx];
						}
					}
				}
			}
			*/
			//以BVH node作为存储
			uint32_t nodes[core_para::BVH_MAX_DEEPTH() +10];
			intersection_result t_res_hit_model;
			cuda_BVH node;
			for (int i = 0; i < mesh_size; i++) {
				cuda_mesh& t_mesh = m_meshs[i];
				int stack_idx = 0;
				nodes[stack_idx++] = 0;//加入根节点,可以用下标代替，以减少栈空间

				while (stack_idx > 0) {
					res.intersect_cnt++;
					node = t_mesh.m_BVH[nodes[--stack_idx]];
					float test_dst = node.m_aabb.intersect(t_ray);
					if (test_dst > res.dist)
						continue;
					if (node.triangle_size != 0) //子节点，BVH只有有两个节点的情况
					{
						for (int tri_idx_offset = 0; tri_idx_offset < node.triangle_size; tri_idx_offset++)
						{//检测三角形
							res.intersect_cnt++;
							cuda_mesh_face& mesh_face = t_mesh.m_face[node.idx + tri_idx_offset];
							cuda_vertex* vertexes = t_mesh.m_vertex;
							t_res_hit_model = ray_triangle_intersect(
								t_ray.get_oirgin(),
								t_ray.get_dir(),
								vertexes[mesh_face.idx[0]].pos,
								vertexes[mesh_face.idx[1]].pos,
								vertexes[mesh_face.idx[2]].pos
							);
							if (t_res_hit_model.hit && t_res_hit_model.t < res.dist) {
								res.is_hit = true;
								res.dist = t_res_hit_model.t;
								res.normal = vertexes[mesh_face.idx[0]].normal * (1.0f - t_res_hit_model.u - t_res_hit_model.v)
									+ vertexes[mesh_face.idx[1]].normal * (t_res_hit_model.u)
									+ vertexes[mesh_face.idx[2]].normal * (t_res_hit_model.v);
								res.position = t_ray.at(t_res_hit_model.t);
								res.uv = vertexes[mesh_face.idx[0]].uv * (1.0f - t_res_hit_model.u - t_res_hit_model.v)
									+ vertexes[mesh_face.idx[1]].uv * (t_res_hit_model.u)
									+ vertexes[mesh_face.idx[2]].uv * (t_res_hit_model.v);
								res.material = m_material;
								res.geo_normal = glm::normalize(//geo normal 可以在模型导入的时候预计算
									glm::cross(vertexes[mesh_face.idx[0]].pos - vertexes[mesh_face.idx[1]].pos,
										vertexes[mesh_face.idx[1]].pos - vertexes[mesh_face.idx[2]].pos
									)
								);
								//纠正
								if (glm::dot(res.normal, t_ray.get_dir()) > 0.0f) //内部纠正
									res.normal *= -1.0f;
								if (glm::dot(res.geo_normal, t_ray.get_dir()) > 0.0f)
									res.geo_normal *= -1.0f;
								//空间变换
								glm::vec3 geo_dir_point = res.position + res.geo_normal;
								glm::vec3 dir_point = res.position + res.normal;
								res.position = *m_inv_translate_matrix * glm::vec4{ res.position,1.0f };
								geo_dir_point = *m_inv_translate_matrix * glm::vec4{ geo_dir_point,1.0f };
								dir_point = *m_inv_translate_matrix * glm::vec4{ dir_point,1.0f };
								
								res.normal = glm::normalize(dir_point - res.position);
								res.geo_normal = glm::normalize(geo_dir_point - res.position);
							}
						}
					}
					else {
						float dst_left = t_mesh.m_BVH[node.idx].m_aabb.intersect(t_ray);
						float dst_right = t_mesh.m_BVH[node.idx + 1].m_aabb.intersect(t_ray);
						uint32_t near_node_idx = dst_left < dst_right ? node.idx : node.idx + 1;
						uint32_t far_node_idx = node.idx + 1 + node.idx - near_node_idx;
						float far_dst = glm::max(dst_left, dst_right);
						float near_dst = dst_left + dst_right - far_dst;

						if (far_dst < res.dist) nodes[stack_idx++] = far_node_idx;
						if (near_dst < res.dist) nodes[stack_idx++] = near_node_idx;

					}
				}
			}
			return res;
		}

	};

}