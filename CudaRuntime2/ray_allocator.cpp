#include "ray_allocator.hpp"
#include<imgui.h>
namespace LuTracer {

	void ray_allocator_displayor::init(int height,int width)
	{
		m_pixel_of_variance_visualize.reset(new uint32_t[height * width]);
		m_pixel_of_sampling_visualize.reset(new uint32_t[height * width]);
		cuda_sampling_data.reset(new float[height * width]);
		m_image_of_variance_visualize = std::make_shared<Walnut::Image>(width, height, Walnut::ImageFormat::RGBA);
		m_image_of_sampling_visualize = std::make_shared<Walnut::Image>(width, height, Walnut::ImageFormat::RGBA);
		m_image_of_variance_visualize.get()->SetData(m_pixel_of_variance_visualize.get());
		m_image_of_sampling_visualize.get()->SetData(cuda_sampling_data.get());
		max_for_width.resize(width);
	}

	void ray_allocator_displayor::show_ui(ray_allocator& alloc)
	{

		std::for_each(std::execution::par, alloc.width_iterator.begin(), alloc.width_iterator.end(),
			[&](size_t pixel_w) {
			for (int pixel_h = 0; pixel_h < alloc.image_height; pixel_h++) {
				int pixel_pos = pixel_h * alloc.image_width + pixel_w;
				float color = std::min(alloc.w_of_per_pixel.get()[pixel_pos] / alloc.max_w, 1.0f);
				unsigned char* pixel = (unsigned char*)&m_pixel_of_variance_visualize.get()[pixel_pos];
				pixel[0] = pixel[1] = pixel[2] = color * 255;
				pixel[3] = 255;
			}
		});

		//选出最大值
		float max_w = -std::numeric_limits<float>::max();
		for (int i = 0; i < max_for_width.size(); i++)
			max_for_width[i] = -std::numeric_limits<float>::max();
		//找到每列的最大值，并计算每个pixel的cost，可能每行有更好的性能
		std::for_each(std::execution::par, alloc.width_iterator.begin(), alloc.width_iterator.end(),
			[&](size_t pixel_w) {
			for (int pixel_h = 0; pixel_h < alloc.image_height; pixel_h++) {
				int pixel_pos = pixel_h * alloc.image_width + pixel_w;
				max_for_width[pixel_w] = std::max(cuda_sampling_data.get()[pixel_pos], max_for_width[pixel_w]);
			}
		});
		for (int i = 0; i < max_for_width.size(); i++)
			max_w = std::max(max_w, max_for_width[i]);

		std::for_each(std::execution::par, alloc.width_iterator.begin(), alloc.width_iterator.end(),
			[&](size_t pixel_w) {
			for (int pixel_h = 0; pixel_h < alloc.image_height; pixel_h++) {
				int pixel_pos = pixel_h * alloc.image_width + pixel_w;
				float color = std::min(cuda_sampling_data.get()[pixel_pos] / max_w, 1.0f);
				unsigned char* pixel = (unsigned char*)&m_pixel_of_sampling_visualize.get()[pixel_pos];
				pixel[0] = pixel[1] = pixel[2] = color * 255;
				pixel[3] = 255;
			}
		});

		m_image_of_variance_visualize->SetData(m_pixel_of_variance_visualize.get());
		m_image_of_sampling_visualize->SetData(m_pixel_of_sampling_visualize.get());
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 0.0f,0.0f });
		ImGui::Begin("ray allocator");

		ImGui::Image(m_image_of_variance_visualize->GetDescriptorSet(), { (float)m_image_of_variance_visualize->GetWidth(),(float)m_image_of_variance_visualize->GetHeight() });
		ImGui::Image(m_image_of_sampling_visualize->GetDescriptorSet(), { (float)m_image_of_variance_visualize->GetWidth(),(float)m_image_of_variance_visualize->GetHeight() });
		ImGui::DragFloat("time weight", &alloc.weight_of_time_elapsed, 0.01f, 0);
		ImGui::DragFloat("variance weight", &alloc.weight_of_radiance_variance, 0.01f, 0);
		ImGui::End();
		ImGui::PopStyleVar();
	}
}
