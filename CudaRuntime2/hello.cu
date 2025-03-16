#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<glm/glm.hpp>
#include "walnut/include/Application.h"
#include "walnut/include/EntryPoint.h"
#include "walnut/include/Image.h"
#include<iostream>
#include<utility.h>
#include<scene.h>
#include<cuda_model.h>
#include<camera.h>
#include<model.h>
#include<Timer.h>
#include<assimp/Importer.hpp>
#include<assimp/scene.h> 
#include<variable.h>
#include<ray_allocator.hpp>
using namespace LuTracer;
cudaError cuda_render(cuda_scene scene);


cudaError joint_bilateral_filtering(joint_bilateral_filter m_filter);


class ExampleLayer : public Walnut::Layer
{

	void cuda_init() {

	}
public:
	int rotate_frame_cnt = 0;
	float fps_mean = 0;
	virtual void OnUIRender() override
	{
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 0.0f,0.0f });

		ImGui::Begin("Viewport");

		if (m_image)
			ImGui::Image(m_image->GetDescriptorSet(), { (float)m_image->GetWidth(),(float)m_image->GetHeight() });
		ImGui::End();
		ImGui::PopStyleVar();

		ImGui::Begin("Scene");
		if (is_use_pixel_importance_sampling)
			ray_allocator_displayor::show_ui(m_ray_allocator);

		ImGui::PushID(std::hash<std::string>{}(std::string("GLOBAL INFORMATION")));
		if (ImGui::ColorEdit3("sky light", &sky_color.x)) {
			ui_change = true;
			check_cf(cudaMemcpy(m_scene.sky_color, &sky_color, sizeof glm::vec3, cudaMemcpyHostToDevice));
		}
		ImGui::PopID();

		ImGui::Checkbox("use filter", &is_use_filter);
		ImGui::Checkbox("use pixel importance", &is_use_pixel_importance_sampling);

		ImGui::Text("FPS_mean: %f", fps_mean / rotate_frame_cnt);
		ImGui::Text("FPS: %f", 1000.0f / (overhead));

		if (ImGui::TreeNode("model information")) {
			for (int model_idx = 0; model_idx < m_models.size(); model_idx++) {
				host_model& m_model = m_models[model_idx];
				ImGui::PushID((int)&m_model);
				if (ImGui::TreeNode(("information of" + m_model.model_name).c_str())) {
					ImGui::Text("leaf_cnt:     %d", m_model.leaf_cnt);
					ImGui::Text("leaf_tri_max: %d", m_model.leaf_tri_max);
					ImGui::Text("leaf_tri_min: %d", m_model.leaf_tri_min);
					ImGui::Text("leaf_tri_mean:%f", m_model.leaf_tri_mean / (float)m_model.leaf_cnt);
					ImGui::Text("max_deepth:   %d", m_model.max_deepth);
					ImGui::Text("min_deppth:   %d", m_model.min_deppth);
					ImGui::Text("deepth_mean:  %f", m_model.deppth_mean / m_model.leaf_cnt);
					ImGui::TreePop();
				}
				ImGui::PopID();
			}
			ImGui::TreePop();
		}
		if (joint_bilateral_filter_displayor::show_ui(m_config)) {
			check_cf(cudaMemcpy(m_scene.m_filter.config, &m_config, sizeof(joint_config), cudaMemcpyHostToDevice));
		}
		for (int model_idx = 0; model_idx < m_models.size(); model_idx++) {
			host_model& m_model = m_models[model_idx];
			cuda_model& cuda_model_ = m_scene.m_models[model_idx];
			model_ui_displayor::display_variation observation = model_ui_displayor::show_ui(m_model);
			if (observation.matrix_flag) {
				//如果矩阵改变，则重新拷贝矩阵
				check_cf(cudaMemcpy(cuda_model_.m_translate_matrix, &m_model.m_translate_matrix, sizeof glm::mat4x4, cudaMemcpyHostToDevice));
				check_cf(cudaMemcpy(cuda_model_.m_inv_translate_matrix, &m_model.m_inv_translate_matrix, sizeof glm::mat4x4, cudaMemcpyHostToDevice));
				ui_change = true;
			}
			if (observation.material_flag) {
				//如果矩阵改变，则重新拷贝矩阵
				check_cf(cudaMemcpy(cuda_model_.m_material, &m_model.m_material, sizeof material, cudaMemcpyHostToDevice));
				ui_change = true;
			}
		}
		ImGui::End();
	}

	virtual void OnUpdate(float ds) override {

		static float time_cnt = .0f;
		time_cnt += ds;


		m_camera->interact(ds);

		if (m_camera->this_frame_is_change()) {
			check_cf(cudaMemcpy(m_scene.camera_ray, m_camera->get_rays(), pixel_size * sizeof LuTracer::ray, cudaMemcpyHostToDevice));
			check_cf(cudaMemcpy(m_scene.pixel_pos, m_default_pixel_pos, pixel_size * sizeof(glm::uvec2), cudaMemcpyHostToDevice));
		}

		if (m_camera->this_frame_is_change() || ui_change) {
			m_fram_cnt = 1;//记录渲染帧数《此时渲染帧数和pixel帧数区分开来》
			std::cout << "change" << std::endl;
			check_cf(cudaMemset(m_scene.pre_pixel_sampling_cnt, 0, pixel_size * sizeof(unsigned int)));
			if (ui_change) ui_change = false;
			check_cf(cudaMemset(m_scene.m_pixel_radiance, 0, m_scene.image_hight * m_scene.image_width * sizeof glm::vec3));
			check_cf(cudaMemset(m_scene.radiance_variance_sum, 0, pixel_size * sizeof(float)));
			check_cf(cudaMemset(m_scene.time_elapsed_sum, 0, pixel_size * sizeof(float)));
		}

		Walnut::Timer timer;
		check_cf(cudaMemcpy(m_scene.frame_cnt, &m_fram_cnt, sizeof(float), cudaMemcpyHostToDevice));

		cuda_render(m_scene);//渲染，同步

		if (is_use_pixel_importance_sampling) {//这个残影bug怎么回事？？
			if (m_fram_cnt % 200 == 0) {
				std::cout << "alloca" << std::endl;
				//拷贝time elapsed 数据
				check_cf(cudaMemcpy(m_ray_allocator.time_elapsed.get(), m_scene.time_elapsed, pixel_size * sizeof(float), cudaMemcpyDeviceToHost));
				//拷贝radiance variance 数据
				check_cf(cudaMemcpy(m_ray_allocator.radiance_variance.get(), m_scene.radiance_variance, pixel_size * sizeof(float), cudaMemcpyDeviceToHost));
				//拷贝每个像素的采样次数
				check_cf(cudaMemcpy(ray_allocator_displayor::cuda_sampling_data.get(), m_scene.pre_pixel_sampling_cnt, pixel_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));

				m_ray_allocator.allocate();
				check_cf(cudaMemcpy(m_scene.camera_ray, m_ray_allocator.m_result_ray.get(), pixel_size * sizeof(ray), cudaMemcpyHostToDevice));
				check_cf(cudaMemcpy(m_scene.pixel_pos, m_ray_allocator.m_result_pixel_pos.get(), pixel_size * sizeof(glm::uvec2), cudaMemcpyHostToDevice));

				cudaDeviceSynchronize();
			}
		}
		else {//这边可以优化一下
			check_cf(cudaMemcpy(m_scene.camera_ray, m_camera->m_rays.get(), pixel_size * sizeof(ray), cudaMemcpyHostToDevice));
			check_cf(cudaMemcpy(m_scene.pixel_pos, m_default_pixel_pos, pixel_size * sizeof(glm::uvec2), cudaMemcpyHostToDevice));
		}

		if (is_use_filter) {
			joint_bilateral_filtering(m_scene.m_filter);
			check_cf(cudaMemcpy(m_scene.m_pixel, m_scene.m_filter.result, pixel_size * pixel_bit, cudaMemcpyDeviceToDevice));
			check_cf(cudaMemset(
				m_scene.m_filter.result,
				0,
				pixel_size * pixel_bit)
			);
		}
		overhead = timer.ElapsedMillis();

		check_cf(cudaMemcpy(m_image_data, m_scene.m_pixel, pixel_size * pixel_bit, cudaMemcpyDeviceToHost));
		m_image->SetData(m_image_data);
		m_fram_cnt++;
	}

	void OnAttach() override {
		cudaSetDevice(0);

		//host端image、camera端的数据更新
			//image
		m_image = std::make_shared<Walnut::Image>(m_viewport_width, m_viewport_height, Walnut::ImageFormat::RGBA);
		m_image_data = new uint32_t[m_viewport_height * m_viewport_width];
		std::memset(m_image_data, 0xffffffff, m_viewport_height * m_viewport_width * sizeof(uint32_t));
		//camera
		m_camera = std::make_shared<LuTracer::camera>(m_image, glm::vec3{ 5,0,1 }, glm::vec3{ -1,0,0 }, 0.5f, 1.0f);
		//pixel size
		pixel_size = m_image->GetHeight() * m_image->GetWidth();
		uint32_t mw = m_image->GetWidth(), mh = m_image->GetHeight();
		int pixel_bit = sizeof uint32_t;

		//m_ray_allocator与camera 指向同一块内存

		m_ray_allocator.init(m_viewport_width, m_viewport_height, m_camera.get()->m_rays);
		ray_allocator_displayor::init(m_viewport_height, m_viewport_width);
		m_default_pixel_pos = new glm::uvec2[pixel_size];
		for (int h = 0; h < mh; h++) {
			for (int w = 0; w < mw; w++) {
				m_default_pixel_pos[h * mw + w] = glm::uvec2{ w,h };
			}
		}

		//cuda全局

		//scene
		m_config.buffer_width = mw;
		m_config.buffer_height = mh;
		m_scene.m_filter.init(mw, mh, m_config);

		int clock_rate_kHz_;
		cudaDeviceGetAttribute(&clock_rate_kHz_, cudaDevAttrClockRate, 0);
		printf("GPU 主频: %d kHz\n", clock_rate_kHz_);
		m_scene.clock_rate_kHz = clock_rate_kHz_;

		check_cf(cudaSetDevice(0));
		check_cf(cudaMalloc((void**)&(m_scene.m_pixel), pixel_size * pixel_bit));
		check_cf(cudaMalloc((void**)&(m_scene.frame_cnt), sizeof(float)));
		check_cf(cudaMalloc((void**)&(m_scene.camera_ray), pixel_size * sizeof LuTracer::ray));
		check_cf(cudaMalloc((void**)&(m_scene.m_pixel_radiance), pixel_size * sizeof glm::vec3));
		check_cf(cudaMalloc((void**)&(m_scene.radiance_variance), pixel_size * sizeof(float)));
		check_cf(cudaMalloc((void**)&(m_scene.radiance_variance_sum), pixel_size * sizeof(float)));
		check_cf(cudaMalloc((void**)&(m_scene.time_elapsed), pixel_size * sizeof(float)));
		check_cf(cudaMalloc((void**)&(m_scene.time_elapsed_sum), pixel_size * sizeof(float)));
		check_cf(cudaMalloc((void**)&(m_scene.pixel_pos), pixel_size * sizeof glm::uvec2));
		check_cf(cudaMalloc((void**)&(m_scene.sky_color), sizeof glm::vec3));
		check_cf(cudaMalloc((void**)&(m_scene.pre_pixel_sampling_cnt), pixel_size * sizeof(unsigned int)));

		m_scene.image_hight = mh;
		m_scene.image_width = mw;

		check_cf(cudaMemcpy(m_scene.m_pixel, m_image_data, pixel_size * pixel_bit, cudaMemcpyHostToDevice));
		m_scene.m_filter.color_buffer = m_scene.m_pixel;
		check_cf(cudaMemcpy(m_scene.camera_ray, m_camera->get_rays(), pixel_size * sizeof LuTracer::ray, cudaMemcpyHostToDevice));
		check_cf(cudaMemcpy(m_scene.pixel_pos, m_default_pixel_pos, pixel_size * sizeof glm::uvec2, cudaMemcpyHostToDevice));
		check_cf(cudaMemcpy(m_scene.sky_color, &sky_color, sizeof glm::vec3, cudaMemcpyHostToDevice));
		check_cf(cudaMemcpy(m_scene.frame_cnt, &m_fram_cnt, sizeof (float), cudaMemcpyHostToDevice));

		check_cf(cudaMemset(m_scene.m_pixel_radiance, 0, pixel_size * sizeof glm::vec3));
		check_cf(cudaMemset(m_scene.pre_pixel_sampling_cnt, 0, pixel_size * sizeof(unsigned int)));
		//model

		//精细度不高时就会出现0个三角形的BVH非叶子节点
		host_model::set_BVH_split_step(20);


		//std::vector<std::string> path = { "./model/AstonMartinDB9_max_mb_lwo_obj/DB9.obj" };
		//std::vector<std::string> path = { "./model/dragon/dragon_65.stl", "./model/bkm/Queen_Nidoqueen.stl" };
		//std::vector<std::string> path = { "./model/dragon/dragon_65.stl" };
		std::vector<std::string> path = {
			"./model/CornellBox/CornellBox_Back.obj",
			"./model/CornellBox/CornellBox_bottom.obj",
			"./model/CornellBox/CornellBox_left.obj",
			"./model/CornellBox/CornellBox_right.obj",
			"./model/CornellBox/CornellBox_top.obj",
			"./model/CornellBox/CornellBox_light.obj",
			"./model/dragon/dragon_65_sm.stl",
			"./model/bkm/Queen_Nidoqueen.stl"
		};
		material mtl;
		mtl.specular = 1.0f;
		mtl.specularTint = 1.0f;
		for (int i = 0; i < path.size(); i++) {
			m_models.push_back({});
			host_model& m_model1 = m_models.back();
			m_model1.set_matrix();
			if (path[i] == "./model/AstonMartinDB9_max_mb_lwo_obj/DB9.obj") {
				mtl.base_color = glm::vec3{ 238,230,133 } / 255.0f;
				mtl.roughness = 1.0f;
				mtl.metallic = 1.0f;
				mtl.emit_light = glm::vec3{ 0,0,0 };
				m_model1.set_model_matrix(glm::vec3{ 1.0f } / 2.0f);
			}
			if (path[i] == "./model/dragon/dragon_65.stl" || path[i] == "./model/dragon/dragon_65_sm.stl") {
				mtl.base_color = glm::vec3{ 0,255,127 } / 255.0f;
				mtl.roughness = 0.3f;
				mtl.metallic = 1.0f;
				mtl.emit_light = glm::vec3{ 0,0,0 };
				m_model1.set_model_matrix(glm::vec3{ 1.0f } / 15.f);
			}
			if (path[i] == "./model/bkm/Queen_Nidoqueen.stl") {
				mtl.base_color = glm::vec3{ 26,25,127 } / 255.0f;
				mtl.roughness = 0.5f;
				mtl.metallic = 0.0f;
				mtl.emit_light = glm::vec3{ 0,0,0 };
				m_model1.set_model_matrix(glm::vec3{ 1.0f } / 75.f);
			}
			if (path[i] == "./model/room_ct/1.obj") {
				mtl.base_color = glm::vec3{ 0,255,127 } / 255.0f;
				mtl.roughness = 1.0f;
				mtl.metallic = 0.0f;
				mtl.emit_light = glm::vec3{ 0,0,0 };
				m_model1.set_model_matrix(glm::vec3{ 1.0f } / 5.f, { 1,0,0 }, 90);
			}
			if (path[i] == "./model/CornellBox/CornellBox_Back.obj") {
				mtl.base_color = glm::vec3{ 255,255,255 } / 255.0f;
				mtl.roughness = 1.0f;
				mtl.metallic = 0.0f;
				mtl.emit_light = glm::vec3{ 0,0,0 };
			}
			if (path[i] == "./model/CornellBox/CornellBox_top.obj") {
				mtl.base_color = glm::vec3{ 255,255,255 } / 255.0f;
				mtl.roughness = 1.0f;
				mtl.metallic = 0.0f;
				mtl.emit_light = glm::vec3{ 0,0,0 };
			}
			if (path[i] == "./model/CornellBox/CornellBox_bottom.obj") {
				mtl.base_color = glm::vec3{ 255,255,255 } / 255.0f;
				mtl.roughness = 1.0f;
				mtl.metallic = 0.0f;
				mtl.emit_light = glm::vec3{ 0,0,0 };
				mtl.emit_light = glm::vec3{ 0,0,0 };
			}
			if (path[i] == "./model/CornellBox/CornellBox_left.obj") {
				mtl.base_color = glm::vec3{ 255,0,0 } / 255.0f;
				mtl.roughness = 1.0f;
				mtl.metallic = 0.0f;
				mtl.emit_light = glm::vec3{ 0,0,0 };
			}
			if (path[i] == "./model/CornellBox/CornellBox_right.obj") {
				mtl.base_color = glm::vec3{ 0,255,0 } / 255.0f;
				mtl.roughness = 1.0f;
				mtl.metallic = 0.0f;
				mtl.emit_light = glm::vec3{ 0,0,0 };
			}
			if (path[i] == "./model/CornellBox/CornellBox_light.obj") {
				mtl.base_color = glm::vec3{ 255,255,255 } / 255.0f;
				mtl.roughness = 1.0f;
				mtl.metallic = 0.0f;
				mtl.emit_light = glm::vec3{ 255,255,255 } / 255.0f;
			}
			if (path[i] == "./model/OBJ/forest.obj" || path[i] == "./model/OBJ/forest_max.obj") {
				mtl.base_color = glm::vec3{ 238,230,133 } / 255.0f;
				mtl.roughness = 1.0f;
				mtl.metallic = 0.0f;
				mtl.emit_light = glm::vec3{ 0,0,0 };
				m_model1.set_model_matrix(glm::vec3{ 1.0f } / 300.0f, { 1,0,0 }, 90);
			}
			m_model1.m_material = mtl;
			m_model1.set_matrix({ 0,0,0 });
			m_model1.load(path[i]);
		}

		m_scene.model_size = m_models.size();

		for (int model_idx = 0; model_idx < m_models.size() && model_idx < core_para::CUDA_MAX_MODEL_NUM(); model_idx++) {
			host_model& m_model = m_models[model_idx];
			cuda_model& d_model = m_scene.m_models[model_idx];

			//分配ray pixel的空间


		//给cuda_model分配内存
			check_cf(cudaMalloc((void**)&(d_model.m_ab_box), sizeof cuda_AABB));
			check_cf(cudaMalloc((void**)&(d_model.m_translate_matrix), sizeof glm::mat4x4));
			check_cf(cudaMalloc((void**)&(d_model.m_inv_translate_matrix), sizeof glm::mat4x4));
			check_cf(cudaMalloc((void**)&(d_model.m_material), sizeof material));

			check_cf(cudaMemcpy(d_model.m_ab_box, &m_model.m_ab_box, sizeof cuda_AABB, cudaMemcpyHostToDevice));
			check_cf(cudaMemcpy(d_model.m_translate_matrix, &m_model.m_translate_matrix, sizeof glm::mat4x4, cudaMemcpyHostToDevice));
			check_cf(cudaMemcpy(d_model.m_inv_translate_matrix, &m_model.m_inv_translate_matrix, sizeof glm::mat4x4, cudaMemcpyHostToDevice));
			check_cf(cudaMemcpy(d_model.m_material, &m_model.m_material, sizeof material, cudaMemcpyHostToDevice));

			//model
			d_model.mesh_size = m_model.m_meshs.size();
			for (int i = 0; i < m_model.m_meshs.size(); i++) {

				int vertex_bit_size = m_model.m_meshs[i].m_vertex.size() * sizeof(cuda_vertex);
				int face_idx_bit_size = m_model.m_meshs[i].m_face.size() * sizeof(cuda_mesh_face);
				int BVH_bit_size = m_model.m_BVH_trees[i].size() * sizeof(cuda_BVH);

				//确定size
				d_model.m_meshs[i].vertex_size = m_model.m_meshs[i].m_vertex.size();
				d_model.m_meshs[i].m_face_size = m_model.m_meshs[i].m_face.size();
				d_model.m_meshs[i].BVH_size = m_model.m_BVH_trees[i].size();

				//分配内存大小
				check_cf(cudaMalloc((void**)&(d_model.m_meshs[i].m_vertex), vertex_bit_size));
				check_cf(cudaMalloc((void**)&(d_model.m_meshs[i].m_face), face_idx_bit_size));
				check_cf(cudaMalloc((void**)&(d_model.m_meshs[i].m_BVH), BVH_bit_size));

				//拷贝数据
				check_cf(cudaMemcpy(d_model.m_meshs[i].m_vertex,
					&m_model.m_meshs[i].m_vertex[0],
					vertex_bit_size,
					cudaMemcpyHostToDevice));
				check_cf(cudaMemcpy(d_model.m_meshs[i].m_face,
					&m_model.m_meshs[i].m_face[0],
					face_idx_bit_size,
					cudaMemcpyHostToDevice));
				check_cf(cudaMemcpy(d_model.m_meshs[i].m_BVH,
					&m_model.m_BVH_trees[i][0],
					BVH_bit_size,
					cudaMemcpyHostToDevice));
			}
		}

	}

	void camera_rotate(glm::vec3 point, float r, float time) {
		float cos_sin_para = time * camera_speed;
		glm::vec3 delta_pos = { glm::cos(cos_sin_para),glm::sin(cos_sin_para) ,0.0f };
		glm::vec3 pos = point + glm::vec3{ delta_pos.x * r,delta_pos.y * r,0.0f };
		glm::vec3 dir = -delta_pos;
		m_camera->set_direction_position(dir, pos);
	}

private:
	std::shared_ptr<LuTracer::camera> m_camera;

	uint32_t m_viewport_height = core_para::IMAGE_HEIGHT(), m_viewport_width = core_para::IMAGE_WIDTH();
	std::shared_ptr<Walnut::Image> m_image;
	uint32_t* m_image_data;
	std::vector<host_model> m_models;

	//uniform data
	uint32_t pixel_size;
	uint32_t pixel_bit = sizeof(uint32_t);
	float overhead = 0;
	float camera_speed = 1.0f;
	glm::vec3 view_point{ 0,0,1 };
	glm::uvec2* m_default_pixel_pos;

	joint_config m_config;
	ray_allocator m_ray_allocator;
	bool is_use_filter = false;
	bool is_use_pixel_importance_sampling = false;


	//cuda_data
	uint32_t* image_width, * image_hight;
	ray* camera_ray;
	uint32_t* m_pixel;
	sphere s1_h{ glm::vec3(0,0,1),0.5 };

	unsigned int m_fram_cnt = 1;
	bool ui_change = false;
	cuda_scene m_scene;
	glm::vec3 sky_color = { 0.5f,0.7f,1.0f };
};

Walnut::Application* Walnut::CreateApplication(int argc, char** argv)
{
	Walnut::ApplicationSpecification spec;
	spec.Name = "Walnut Example";

	Walnut::Application* app = new Walnut::Application(spec);
	app->PushLayer<ExampleLayer>();
	app->SetMenubarCallback([app]()
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Exit"))
			{
				app->Close();
			}
			ImGui::EndMenu();
		}
	});
	return app;
}



int main()
{
	Walnut::Main(0, nullptr);
	return 0;
}

