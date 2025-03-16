#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<host_AABB.h>
#include<ray.h>
#include<host_BVH.hpp>
#include<core_para.h>
#include<material.hpp>

#include<glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp> // 包含矩阵变换函数

#include<imgui/imgui.h>

#include<assimp/Importer.hpp>
#include<assimp/scene.h>
#include <assimp/postprocess.h>

#include<iostream>
#include<limits>

namespace LuTracer {

	struct g_hit_payload {
		bool is_hit = false;
		float dist;
		ray next_ray;
		glm::vec4 emit_color = { 0,0,0,0 };
		glm::vec2 uv;
		glm::vec3 geo_normal;
		glm::vec3 normal;
		glm::vec3 position;
	};

	class sphere {
		glm::vec3 pos;
		float radius;
	public:
		__host__ __device__ sphere(glm::vec3 position, float radius) :
			pos(position), radius(radius) {
		}
		__host__ __device__ sphere() = default;
		__host__ __device__ glm::vec3& get_position() { return pos; }

		__host__ __device__ void set_position(const glm::vec3& pos);
		__host__ __device__ void offset_position(const glm::vec3 pos);
		__host__ __device__ void rotate();

		__host__ __device__ float get_radius() { return radius; }

		__host__ __device__ g_hit_payload hit_test(const ray& r) {
			//t2d⋅d−2td⋅(C−Q)+(C−Q)⋅(C−Q)−r2=0,解方程
			g_hit_payload payload;
			auto& d = r.get_dir();
			auto& Q = r.get_oirgin();
			glm::vec3& C = pos;
			float a = glm::dot(d, d);
			float b = -2.0f * glm::dot(d, C - Q);
			float c = glm::dot(C - Q, C - Q) - radius * radius;
			float dertermine = b * b - 4 * a * c;
			if (dertermine < 0) {
				payload.is_hit = false;
				return payload;
			}
			float vert_a = (1.0f / a) * 0.5f;
			float sq_dertermine = glm::sqrt(dertermine);
			float solve_1 = (-b + sq_dertermine) * vert_a;
			if (solve_1 < 0.0f) {
				payload.is_hit = false;
				return payload;
			}
			payload.is_hit = true;
			float solve_2 = (-b - sq_dertermine) * vert_a;

			if (solve_2 > 0.0f) {
				payload.dist = solve_2;
			}
			else {
				payload.dist = solve_1;
			}
			payload.position = r.at(payload.dist);
			payload.geo_normal = glm::normalize(payload.position - pos);
			if (glm::dot(payload.geo_normal, r.get_dir()) > 0.0f) {//法向和光线方向点乘大于0：在内部
				payload.geo_normal *= -1;
			}
			return payload;
		}

	};



	struct host_vertex {
		glm::vec3 pos, normal;
		glm::vec2 uv{ 0,0 };
		glm::vec3 tangent, bitangent;
	};

	struct host_mesh_face {
		uint32_t idx[3];
	};

	struct host_mesh {
		std::vector<host_vertex> m_vertex;
		std::vector<host_mesh_face> m_face;//bvh可能要重构m_face以满足bvh要求
		host_AABB mesh_box;
	};
	//有将所有mesh变成一个mesh的操作，后期修改

	struct host_material {
		glm::vec4 base_color;
		float roughness;
		float metallic;
		float specular = 1.f;
		float specularTint = 0.0f;
	};

	struct host_model {
	private:
		bool is_set_mat = false;
		inline static uint32_t axis_split_step = 1000;
		//一些统计数据
	public:

		int leaf_tri_min = std::numeric_limits<int>::max();
		int leaf_tri_max = std::numeric_limits<int>::min();
		float leaf_tri_mean = 0;
		int leaf_cnt = 0;
		int max_deepth = std::numeric_limits<int>::min();
		int min_deppth = std::numeric_limits<int>::max();
		float deppth_mean = 0;

		std::vector<std::vector<host_BVH>> m_BVH_trees;
		std::vector<host_mesh> m_meshs;

		//变换信息
		glm::mat4x4 m_translate_matrix{ 1.0f };
		glm::mat4x4 m_inv_translate_matrix{ 1.0f };
		glm::vec3 translation{ 0,0,0 };
		glm::vec3 scale{ 1,1,1 };
		glm::vec3 rotation_axis{ 0,0,1 };
		float rotation_angle = 0.0f;

		//加载时的变换矩阵
		glm::mat3x3 m_model_load_mat{ 1.0f };
		host_AABB m_ab_box;

		//材质
		material m_material;

		std::string m_directory;
		std::string model_name;


		static glm::mat4 create_model_matrix(
			glm::vec3 translation,   // 位移 (tx, ty, tz)
			glm::vec3 scale,         // 缩放 (sx, sy, sz)
			float rotationAngle,     // 旋转角度（单位：度）
			glm::vec3 rotationAxis   // 旋转轴 (如绕Y轴旋转是 (0,1,0))
		) {
			// 初始化单位矩阵
			glm::mat4 model = glm::mat4(1.0f);

			// 应用缩放
			model = glm::translate(model, translation);
			float radians = glm::radians(rotationAngle);
			model = glm::rotate(model, radians, rotationAxis);
			model = glm::scale(model, scale);
			// 应用旋转（将角度转换为弧度）
			// 应用平移

			return model;
		}

		static void set_BVH_split_step(uint32_t step_num) {
			axis_split_step = step_num;
		}

		//mat*vec != vec* mat
		void set_matrix(glm::vec3 translation = { 0,0,0 }, glm::vec3 scale = { 1,1,1 }, glm::vec3 rotationAxis = { 0,0,1 }, float rotationAngle = 0.0f) {
			this->translation = translation;
			this->scale = scale;
			this->rotation_axis = rotationAxis;
			this->rotation_angle = rotationAngle;
			matrix_updata();
		}

		void matrix_updata() {
			m_inv_translate_matrix = create_model_matrix(translation, scale, rotation_angle, rotation_axis);
			m_translate_matrix = glm::inverse(m_inv_translate_matrix);
		}

		void set_model_matrix(glm::vec3 scale = { 1,1,1 }, glm::vec3 rotationAxis = { 0,0,1 }, float rotationAngle = 0.0f) {
			m_model_load_mat = create_model_matrix({ 0,0,0 }, scale, rotationAngle, rotationAxis);
		}

		const host_AABB& get_aabb() { return m_ab_box; }
		void load(const std::string& path) {
			Assimp::Importer importer;
			const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
			if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // 如果不是0
			{
				std::cout << "错误::ASSIMP:: " << importer.GetErrorString() << std::endl;
				return;
			}
			m_directory = path.substr(0, path.find_last_of('/'));
			model_name = path.substr(path.find_last_of('/'), path.find_last_of('.'));
			m_meshs.push_back({});
			process_node(scene->mRootNode, scene);
			build_BVH(m_meshs.back());
			test_bvh();
		}

		void test_bvh() {
			auto& tree = m_BVH_trees[0];
			host_BVH& root = tree[0];
			int idx = 0;
			int cnt = 0;
			host_BVH stack[core_para::BVH_MAX_DEEPTH() * 10 + 10];
			stack[idx++] = root;
			while (idx) {
				cnt++;
				host_BVH nd = stack[--idx];
				if (nd.triangle_size == 0) {
					stack[idx++] = tree[nd.idx + 1];
					stack[idx++] = tree[nd.idx];
				}
			}
			if (cnt == tree.size())
				printf("bvh is ok\n");
		}

		void process_node(aiNode* node, const aiScene* scene) {
			for (int i = 0; i < node->mNumMeshes; i++) {
				aiMesh* a_mesh = scene->mMeshes[node->mMeshes[i]];
				process_mesh(a_mesh, scene);
			}

			for (unsigned int i = 0; i < node->mNumChildren; i++) {
				process_node(node->mChildren[i], scene);
			}
		}

		void process_mesh(aiMesh* a_mesh, const aiScene* scene) {
			std::cout << "process_mesh" << std::endl;
			auto& host_mesh = m_meshs.back();
			//host_mesh.m_vertex.resize(a_mesh->mNumVertices);
			//host_mesh.m_face.resize(a_mesh->mNumFaces);
			int mesh_begin_vertex_idx = host_mesh.m_vertex.size();
			for (int i = 0; i < a_mesh->mNumVertices; i++) {
				host_mesh.m_vertex.push_back({});
				//加载顶点
				auto& a_vertex = a_mesh->mVertices[i];
				host_mesh.m_vertex[mesh_begin_vertex_idx + i].pos = glm::vec3{ a_vertex.x,a_vertex.y, a_vertex.z };
				//空间变换
				host_mesh.m_vertex[mesh_begin_vertex_idx + i].pos = m_model_load_mat * host_mesh.m_vertex[mesh_begin_vertex_idx + i].pos;
				//AABB实现
				m_ab_box.insert(host_mesh.m_vertex[mesh_begin_vertex_idx + i].pos);
				host_mesh.mesh_box.insert(host_mesh.m_vertex[mesh_begin_vertex_idx + i].pos);
				//加载法线
				auto& a_vertex_normal = a_mesh->mNormals[i];
				host_mesh.m_vertex[mesh_begin_vertex_idx + i].normal = m_model_load_mat * glm::vec3{ a_vertex_normal.x,a_vertex_normal.y, a_vertex_normal.z };
				//加载uv,假设mesh只有一个纹理，否则用while
				if (a_mesh->mTextureCoords[0]) // 网格是否包含纹理坐标？
					// 顶点最多可包含8个不同的纹理坐标。 因此，我们假设我们不会使用顶点可以具有多个纹理坐标的模型，因此我们总是采用第一个集合（0）。
					host_mesh.m_vertex[mesh_begin_vertex_idx + i].uv = { a_mesh->mTextureCoords[0][i].x,a_mesh->mTextureCoords[0][i].y };
				else
					host_mesh.m_vertex[mesh_begin_vertex_idx + i].uv = { 0,0 };

				//切线
				if (a_mesh->mTangents)
					host_mesh.m_vertex[mesh_begin_vertex_idx + i].tangent = m_model_load_mat * glm::vec3{ a_mesh->mTangents[i].x,a_mesh->mTangents[i].y,a_mesh->mTangents[i].z };
				if (a_mesh->mBitangents)
					host_mesh.m_vertex[mesh_begin_vertex_idx + i].bitangent = m_model_load_mat * glm::vec3{ a_mesh->mBitangents[i].x,a_mesh->mBitangents[i].y,a_mesh->mBitangents[i].z };
			}
			//加载顶点索引
			uint32_t mesh_begin_face_idx = host_mesh.m_face.size();
			for (int i = 0; i < a_mesh->mNumFaces; i++) {
				host_mesh.m_face.push_back({});
				aiFace& a_face = a_mesh->mFaces[i];
				//目前只考虑三角面片
				for (int j = 0; j < 3; j++) {
					host_mesh.m_face[mesh_begin_face_idx + i].idx[j] = a_face.mIndices[j] + mesh_begin_vertex_idx;
				}
			}
			//加载纹理

			//构建mesh的BVH
		}

		void build_BVH(host_mesh& t_mesh) {

			m_BVH_trees.push_back({});//建立树
			auto& tree = m_BVH_trees.back();//建立根节点
			tree.push_back({});
			host_BVH& root = tree.back();
			root.m_aabb = t_mesh.mesh_box;

			//创建new_mesh_face
			std::vector<host_mesh_face> new_mesh_face;
			new_mesh_face.resize(t_mesh.m_face.size());
			uint32_t idx = 0;

			float cost = root.m_aabb.get_area() * t_mesh.m_face.size();

			split(0, t_mesh.m_face, tree, new_mesh_face, idx, t_mesh, cost, 0);

			t_mesh.m_face = new_mesh_face;
		}

		void choose_split(host_BVH& node, std::vector<host_mesh_face>& triangles, host_mesh& t_mesh,
			uint32_t& axis, float& pos, float& cost) {
			float cost_res = std::numeric_limits<float>::max();

			for (int axis_res = 0; axis_res < 3; axis_res++) {
				host_AABB left_ab, right_ab;
				float begin_pos = node.m_aabb.min[axis_res];
				float len = node.m_aabb.max[axis_res] - node.m_aabb.min[axis_res];
				for (int split_step = 0; split_step < axis_split_step; split_step++) {
					float pos_res = begin_pos + (len / axis_split_step) * (split_step + 1);
					uint32_t in_left_num = 0, in_right_num = 0;

					for (auto& tri : triangles) {
						glm::vec3 center = (t_mesh.m_vertex[tri.idx[0]].pos +
							t_mesh.m_vertex[tri.idx[1]].pos +
							t_mesh.m_vertex[tri.idx[2]].pos) / 3.0f;
						if (center[axis_res] < pos_res) {
							left_ab.insert(t_mesh.m_vertex[tri.idx[0]].pos);
							left_ab.insert(t_mesh.m_vertex[tri.idx[1]].pos);
							left_ab.insert(t_mesh.m_vertex[tri.idx[2]].pos);
							in_left_num++;
						}
						else {
							right_ab.insert(t_mesh.m_vertex[tri.idx[0]].pos);
							right_ab.insert(t_mesh.m_vertex[tri.idx[1]].pos);
							right_ab.insert(t_mesh.m_vertex[tri.idx[2]].pos);
							in_right_num++;
						}
					}
					float new_cost = left_ab.get_area() * in_left_num + right_ab.get_area() * in_right_num;
					if (new_cost < cost_res && in_left_num && in_right_num) {
						cost_res = new_cost;
						cost = cost_res;
						pos = pos_res;
						axis = axis_res;
					}
				}
			}
		}

		void split(
			uint32_t bvh_node_idx,//当前的bvh
			std::vector<host_mesh_face>& triangles,//当前bvh需要负责的face
			std::vector<host_BVH>& tree,
			std::vector<host_mesh_face>& new_mesh_face,
			uint32_t& new_mesh_face_idx,
			host_mesh& t_mesh,
			float parent_cost,
			int deepth)
		{
			//超过最大深度时或者三角形数量很少时return
			if (deepth > core_para::BVH_MAX_DEEPTH() || triangles.size() < core_para::BVH_MAX_FACE()) {
				tree[bvh_node_idx].idx = new_mesh_face_idx;
				tree[bvh_node_idx].triangle_size = triangles.size();
				for (int ts = 0; ts < tree[bvh_node_idx].triangle_size; ts++) {
					new_mesh_face[new_mesh_face_idx++] = triangles[ts];
				}
				max_deepth = std::max(deepth, max_deepth);
				min_deppth = std::min(deepth, min_deppth);
				leaf_cnt++;
				leaf_tri_max = std::max((int)triangles.size(), leaf_tri_max);
				leaf_tri_min = std::min((int)triangles.size(), leaf_tri_min);
				leaf_tri_mean += triangles.size();
				deppth_mean += deepth;
				return;
			}
			//分割
			std::vector<host_mesh_face> left_tris, right_tris;
			host_AABB left_ab, right_ab;

			uint32_t axis_idx;
			float split_pos, cost;
			choose_split(tree[bvh_node_idx], triangles, t_mesh, axis_idx, split_pos, cost);
			//考虑parent cost
			/*
			if (parent_cost < cost) {
				tree[bvh_node_idx].idx = new_mesh_face_idx;
				tree[bvh_node_idx].triangle_size = triangles.size();
				for (int ts = 0; ts < tree[bvh_node_idx].triangle_size; ts++) {
					new_mesh_face[new_mesh_face_idx++] = triangles[ts];
				}
				max_deepth = std::max(deepth, max_deepth);
				min_deppth = std::min(deepth, min_deppth);
				leaf_cnt++;
				leaf_tri_max = std::max((int)triangles.size(), leaf_tri_max);
				leaf_tri_min = std::min((int)triangles.size(), leaf_tri_min);
				leaf_tri_mean += triangles.size();
				deppth_mean += deepth;
				return;
			}
			*/
			for (auto& tri : triangles) {
				glm::vec3 center = (t_mesh.m_vertex[tri.idx[0]].pos
					+ t_mesh.m_vertex[tri.idx[1]].pos
					+ t_mesh.m_vertex[tri.idx[2]].pos) / 3.0f;

				if (center[axis_idx] < split_pos) {
					left_ab.insert(t_mesh.m_vertex[tri.idx[0]].pos);
					left_ab.insert(t_mesh.m_vertex[tri.idx[1]].pos);
					left_ab.insert(t_mesh.m_vertex[tri.idx[2]].pos);
					left_tris.push_back(tri);
				}
				else {
					right_ab.insert(t_mesh.m_vertex[tri.idx[0]].pos);
					right_ab.insert(t_mesh.m_vertex[tri.idx[1]].pos);
					right_ab.insert(t_mesh.m_vertex[tri.idx[2]].pos);
					right_tris.push_back(tri);
				}

			}
			//这里有一个优化：left_child_idx + 1 == right_child_idx
			//要先push，再引用如果是push 引用 push 引用，则第二次引用的时候前一个引用会失效
			tree.push_back({});
			tree.push_back({});

			host_BVH& left_node = tree[tree.size() - 2];
			left_node.m_aabb = left_ab;

			host_BVH& right_node = tree[tree.size() - 1];
			right_node.m_aabb = right_ab;

			if (left_tris.size() == 0 || right_tris.size() == 0) {
				std::cout << "error bvh is wrong";
			}

			tree[bvh_node_idx].idx = tree.size() - 2;

			//递归放下面，这样的话父子节点会挨在一起，这样cache命中率高
			float cost_left = left_ab.get_area() * left_tris.size();
			float cost_right = right_ab.get_area() * right_tris.size();

			if (left_tris.size())
				split(tree[bvh_node_idx].idx,
					left_tris, tree,
					new_mesh_face,
					new_mesh_face_idx,
					t_mesh,
					cost_left,
					deepth + 1);

			if (right_tris.size())
				split(tree[bvh_node_idx].
					idx + 1,
					right_tris, tree,
					new_mesh_face,
					new_mesh_face_idx,
					t_mesh,
					cost_right,
					deepth + 1);
		}

	};


	struct model_ui_displayor {
		struct display_variation {
			bool matrix_flag = false;
			bool material_flag = false;
		};
		static display_variation show_ui(host_model& t_model) {
			ImGui::PushID(std::hash<std::string>{}(t_model.model_name));
			display_variation res;
			if (ImGui::TreeNode(t_model.model_name.c_str())) {
				res.matrix_flag |= ImGui::DragFloat3("translation", &t_model.translation.x, 0.01f);
				res.matrix_flag |= ImGui::DragFloat3("scale", &t_model.scale.x, 0.01f);
				res.matrix_flag |= ImGui::DragFloat3("rotation axis", &t_model.rotation_axis.x, 0.01f);
				res.matrix_flag |= ImGui::DragFloat("rotation angle", &t_model.rotation_angle, 0.01f);
				if (res.matrix_flag)
					t_model.matrix_updata();
				res.material_flag |= ImGui::DragFloat("roughness", &t_model.m_material.roughness, 0.01f);
				res.material_flag |= ImGui::DragFloat("metallic", &t_model.m_material.metallic, 0.01f);
				res.material_flag |= ImGui::ColorEdit3("base_color", &t_model.m_material.base_color.x);
				res.material_flag |= ImGui::ColorEdit3("emit_light", &t_model.m_material.emit_light.x);
				res.material_flag |= ImGui::DragFloat("light strength", &t_model.m_material.emit_light_strength, 0.01f);

				ImGui::TreePop();
			}
			ImGui::PopID();
			return res;
		}
	};
}