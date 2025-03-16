#include<joint_bilateral_filter.h>
#include<imgui/imgui.h>
namespace LuTracer {
	bool joint_bilateral_filter_displayor::show_ui(joint_config& m_filter) {
		bool flag = false;
		ImGui::PushID((int)&m_filter);
		if (ImGui::TreeNode("filter config")) {
			int kerbel_size = m_filter.kernel_size;
			flag |= ImGui::DragInt("kernel size", &kerbel_size, 0, 20);
			m_filter.kernel_size = kerbel_size;
			flag |= ImGui::DragFloat("sigma_normal", &m_filter.sigma_normal, 0.001f, 0.f, 100.f);
			flag |= ImGui::DragFloat("sigma_position", &m_filter.sigma_position, 0.001f, 0.f, 100.f);
			flag |= ImGui::DragFloat("sigma_albedo", &m_filter.sigma_albedo, 0.001f, 0.f, 100.f);
			flag |= ImGui::DragFloat("sigma_roughness", &m_filter.sigma_roughness, 0.001f, 0.f, 100.f);
			flag |= ImGui::DragFloat("sigma_color", &m_filter.sigma_color, 0.001f, 0.f, 100.f);
			ImGui::TreePop();
		}
		ImGui::PopID();
		return flag;
	}
}