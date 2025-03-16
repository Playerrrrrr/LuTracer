#pragma once
#include<memory>
#include<ray.h>
#include<share.h>
#include<vector>
#include<execution>
#include<atomic>
#include<Random.h>
#include<Image.h>

#include<iostream>
namespace LuTracer {

	struct ray_allocator_displayor;

	class ray_allocator {
	private:
		std::shared_ptr<ray> camera_ray;//这个是用camera里的那块内存
		std::vector<size_t> width_iterator;
		std::vector<float> max_for_width;
		std::shared_ptr<float> w_of_per_pixel;

		int image_width, image_height;
		float max_w;
		int pixel_size;
		std::atomic<int> idx;
		float cost_fuction(float time,float var) {
			float w_time = weight_of_time_elapsed * glm::atan(time);
			float w_radiance = weight_of_radiance_variance * glm::atan(var);
			return w_time + w_radiance;
		}

		friend struct ray_allocator_displayor;

	public:
		std::shared_ptr<ray> m_result_ray;
		std::shared_ptr<float> time_elapsed;
		std::shared_ptr<glm::uvec2> m_result_pixel_pos;
		std::shared_ptr<float> radiance_variance;
		float weight_of_radiance_variance = 1.0f,weight_of_time_elapsed = 1.0f;
		float* w_of_per_pixel_cuda_filter;

		void init(int width,int height,std::shared_ptr<ray> rays) {
			image_width = width;
			image_height = height;

			pixel_size = image_height * image_width;
			width_iterator.resize(width);
			max_for_width.resize(width);

			time_elapsed.reset(new float[pixel_size]);
			radiance_variance.reset(new float[pixel_size]);
			w_of_per_pixel.reset(new float[pixel_size]);
			m_result_ray.reset(new ray[pixel_size]);
			m_result_pixel_pos.reset(new glm::uvec2[pixel_size]);

			camera_ray = rays;

			for (int i = 0; i < width_iterator.size(); i++)
				width_iterator[i] = i;

			//内存初始化
			std::memset(w_of_per_pixel.get(), 0, sizeof(float) * pixel_size);
		}
		void allocate() {
			//计算最大值
			max_w = -std::numeric_limits<float>::max();
			for (int i = 0; i < max_for_width.size(); i++)
				max_for_width[i] = -std::numeric_limits<float>::max();
			//找到每列的最大值，并计算每个pixel的cost，可能每行有更好的性能
			std::for_each(std::execution::par, width_iterator.begin(), width_iterator.end(),
				[&, this](size_t pixel_w) {
				for (int pixel_h = 0; pixel_h < image_height; pixel_h++) {
					float w = cost_fuction(
						time_elapsed.get()[pixel_h * image_width + pixel_w],
						radiance_variance.get()[pixel_h * image_width + pixel_w]
					);
					w_of_per_pixel.get()[pixel_h * image_width + pixel_w] = w;
					max_for_width[pixel_w] = std::max(w, max_for_width[pixel_w]);
				}
			});
			for (int i = 0; i < max_for_width.size(); i++)
				max_w = std::max(max_w, max_for_width[i]);
			std::cout << max_w << std::endl;
			//用舍选法找到分布

			idx = 0;
			std::for_each(std::execution::par, width_iterator.begin(), width_iterator.end(),
				[&, this](size_t pixel_w) {
				while (idx < pixel_size) {
					float rand1 = Walnut::Random::Float(), rand2 = Walnut::Random::Float();
					float rand3 = Walnut::Random::Float();
					int pixel_x = std::min((int)(rand1 * image_width), image_width - 1);
					int pixel_y = std::min((int)(rand2 * image_height), image_height - 1);
					float rand_val = rand3 * max_w;
					float w_of_pixel = w_of_per_pixel.get()[pixel_y * image_width + pixel_x];
					if (rand_val < w_of_pixel) {
						int t = idx++;
						int pixel_pos = std::min(t, pixel_size - 1);
						m_result_pixel_pos.get()[pixel_pos] = { pixel_x,pixel_y };
						m_result_ray.get()[pixel_pos] = camera_ray.get()[pixel_y * image_width + pixel_x];
					}
				}
			});

		}
	};

	struct ray_allocator_displayor {
	private:
		inline static std::shared_ptr<uint32_t> m_pixel_of_variance_visualize;
		inline static std::shared_ptr<uint32_t> m_pixel_of_sampling_visualize;
		inline static std::shared_ptr<Walnut::Image> m_image_of_variance_visualize;
		inline static std::shared_ptr<Walnut::Image> m_image_of_sampling_visualize;
		inline static std::vector<float> max_for_width;
	public:
		inline static std::shared_ptr<float> cuda_sampling_data;
		static void init(int height, int width);
		static void show_ui(ray_allocator&);
	};

}