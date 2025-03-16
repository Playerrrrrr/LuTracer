#pragma once
#include<memory>
#include<glm/glm.hpp>
#include<Image.h>
#include<ray.h>
#include<iostream>
#include<Input/input.h>
#include<algorithm>
#include<vector>
#include<execution>

class ExampleLayer;
namespace LuTracer {
	class camera {
		const glm::vec3 sky_top{ 0,0,1.0 };
		glm::vec3 pos;
		glm::vec3 dir, right, up;
		glm::vec2 viewport;
		glm::vec3 viewport_u, viewport_v;
		glm::vec2 viewport_delta;//[0]为u_delta,[1]为v_delta
		glm::vec3 pixel00_loc;
		float aspect_ration;
		float near, far;
		bool is_change = false;
		std::shared_ptr<ray> m_rays;
		std::vector<size_t> width_iterator;


		glm::vec2 mouse_last_position;
		float move_speed = 5.0f;


		//这个image不应该被依赖，以后修改过来
		std::shared_ptr<Walnut::Image> image;

		friend class ::ExampleLayer;

		ray cul_ray(glm::ivec2 co_pos) {
			ray res;
			res.origin = pos;
			glm::vec3 pixel_loc = pixel00_loc + co_pos[0] * viewport_delta[0] * right - co_pos[1] * viewport_delta[1] * up;
			res.dir = pixel_loc - pos;
			res.dir = glm::normalize(res.dir);
			return { res.origin,res.dir};
		}

		void update() {
			viewport_delta = { viewport[0] / image->GetWidth(),viewport[1] / image->GetHeight() };
			//计算右边
			right = glm::cross(sky_top, dir); right = glm::normalize(right);
			up = glm::cross(dir, right); up = glm::normalize(up);
			glm::vec3 plate_center = pos + dir * near;
			//计算新uv
			viewport_u = right * viewport[0];
			viewport_v = -up * viewport[1];
			//计算第一个像素的空间位置
			pixel00_loc = plate_center - viewport_u * 0.5f - viewport_v * 0.5f;
			std::for_each(std::execution::par, width_iterator.begin(), width_iterator.end(),
				[&, this](size_t pixel_w) {
				for (int pixel_h = 0; pixel_h < image->GetHeight(); pixel_h++) {
					m_rays.get()[pixel_h * image->GetWidth() + pixel_w] = cul_ray({ pixel_w ,pixel_h });
				}
			});
		}
		

	public:

		camera(std::shared_ptr<Walnut::Image> image, glm::vec3 position, glm::vec3 direction, float hight, float near = 1, float far = 100)
			:aspect_ration(image->GetWidth() / (float)image->GetHeight()), near(near), far(far),
			pos(position), dir(glm::normalize(direction)), image(image) {

			width_iterator = std::vector<size_t>(image->GetWidth());
			viewport = glm::vec2{ aspect_ration * hight,hight };
			for (int i = 0; i < image->GetWidth(); i++) width_iterator[i] = i;
			m_rays.reset(new ray[image->GetHeight() * image->GetWidth()]);
			update();

		}
		//co_pos :width height
		//缓存ray优化 
		ray get_ray(glm::ivec2 co_pos) {
			return m_rays.get()[co_pos.x + co_pos.y * image->GetWidth()];
		}
		ray get_ray(glm::vec2 co_pos) {
			return get_ray(glm::ivec2{ co_pos[0] * image->GetWidth() ,co_pos[1] * image-> GetHeight() });
		}

		ray* get_rays() { return m_rays.get(); }

		std::shared_ptr<ray> get_rays_of_shared() {
			return m_rays;
		}

		void interact(float delta_time) {
			glm::vec2 mouse_pos = Walnut::Input::GetMousePosition();
			glm::vec2 delta = (mouse_pos - mouse_last_position) * 0.002f;
			mouse_last_position = mouse_pos;
			is_change = false;
			if (!Walnut::Input::IsMouseButtonDown(Walnut::MouseButton::Right)) {
				Walnut::Input::SetCursorMode(Walnut::CursorMode::Normal);
				return;
			}
			is_change = true;
			Walnut::Input::SetCursorMode(Walnut::CursorMode::Locked);
			//更新方向
			dir += right * delta.x;
			dir -= up * delta.y;
			dir = glm::normalize(dir);

			bool is_move = false;
			
			if (Walnut::Input::IsKeyDown(Walnut::KeyCode::W)) {
				pos += dir * delta_time * move_speed; 
				is_move = true;
			}
			else if(Walnut::Input::IsKeyDown(Walnut::KeyCode::S)) {
				pos -= dir * delta_time * move_speed;
				is_move = true;
			}
			else if(Walnut::Input::IsKeyDown(Walnut::KeyCode::A)) {
				pos -= right * delta_time * move_speed;
				is_move = true;
			}else if(Walnut::Input::IsKeyDown(Walnut::KeyCode::D)) {
				pos += right * delta_time * move_speed;
				is_move = true;
			}
			update();
		}

		void set_pos(glm::vec3&& position) {
			this->pos = position;
			update();
		}
		void set_near(float near) {
			this->near = near;
			update();

		}
		void set_direction(glm::vec3&& direction){
			this->dir = glm::normalize(direction);
			update();
		}

		void set_direction_position(glm::vec3 dir, glm::vec3 pos) {
			this->dir = dir;
			this->pos = pos;
			is_change = true;
			update();
		}

		bool this_frame_is_change() { return is_change; }
	};
}