#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<glm/glm.hpp>
#include<variable.h>


namespace LuTracer {

	cudaError check_cf_(cudaError t, const char* filename, int line);

#define check_cf(x) check_cf_(x,__FILE__,__LINE__);

#define scope_begin(times,functions)\
        cudaEvent_t start##times,end##times;\
        check_cf(cudaEventCreate(&start##times))\
        check_cf(cudaEventCreate(&end##times))\
        check_cf(cudaEventRecord(start##times))\
        cudaEventQuery(start##times);\
        functions\
        check_cf(cudaEventRecord(end##times))\
        check_cf(cudaEventSynchronize(end##times))\
        check_cf(cudaEventElapsedTime(&times,start##times,end##times))\
        check_cf(cudaEventDestroy(start##times))\
        check_cf(cudaEventDestroy(end##times))


	using  uint = unsigned int;

	__constant__ static const uint soboal_V[8 * 32] = {
2147483648, 1073741824, 536870912, 268435456, 134217728, 67108864, 33554432, 16777216, 8388608, 4194304, 2097152, 1048576, 524288, 262144, 131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1,
2147483648, 3221225472, 2684354560, 4026531840, 2281701376, 3422552064, 2852126720, 4278190080, 2155872256, 3233808384, 2694840320, 4042260480, 2290614272, 3435921408, 2863267840, 4294901760, 2147516416, 3221274624, 2684395520, 4026593280, 2281736192, 3422604288, 2852170240, 4278255360, 2155905152, 3233857728, 2694881440, 4042322160, 2290649224, 3435973836, 2863311530, 4294967295,
2147483648, 3221225472, 1610612736, 2415919104, 3892314112, 1543503872, 2382364672, 3305111552, 1753219072, 2629828608, 3999268864, 1435500544, 2154299392, 3231449088, 1626210304, 2421489664, 3900735488, 1556135936, 2388680704, 3314585600, 1751705600, 2627492864, 4008611328, 1431684352, 2147543168, 3221249216, 1610649184, 2415969680, 3892340840, 1543543964, 2382425838, 3305133397,
2147483648, 3221225472, 536870912, 1342177280, 4160749568, 1946157056, 2717908992, 2466250752, 3632267264, 624951296, 1507852288, 3872391168, 2013790208, 3020685312, 2181169152, 3271884800, 546275328, 1363623936, 4226424832, 1977167872, 2693105664, 2437829632, 3689389568, 635137280, 1484783744, 3846176960, 2044723232, 3067084880, 2148008184, 3222012020, 537002146, 1342505107,
2147483648, 1073741824, 536870912, 2952790016, 4160749568, 3690987520, 2046820352, 2634022912, 1518338048, 801112064, 2707423232, 4038066176, 3666345984, 1875116032, 2170683392, 1085997056, 579305472, 3016343552, 4217741312, 3719483392, 2013407232, 2617981952, 1510979072, 755882752, 2726789248, 4090085440, 3680870432, 1840435376, 2147625208, 1074478300, 537900666, 2953698205,
2147483648, 1073741824, 1610612736, 805306368, 2818572288, 335544320, 2113929216, 3472883712, 2290089984, 3829399552, 3059744768, 1127219200, 3089629184, 4199809024, 3567124480, 1891565568, 394297344, 3988799488, 920674304, 4193267712, 2950604800, 3977188352, 3250028032, 129093376, 2231568512, 2963678272, 4281226848, 432124720, 803643432, 1633613396, 2672665246, 3170194367,
2147483648, 3221225472, 2684354560, 3489660928, 1476395008, 2483027968, 1040187392, 3808428032, 3196059648, 599785472, 505413632, 4077912064, 1182269440, 1736704000, 2017853440, 2221342720, 3329785856, 2810494976, 3628507136, 1416089600, 2658719744, 864310272, 3863387648, 3076993792, 553150080, 272922560, 4167467040, 1148698640, 1719673080, 2009075780, 2149644390, 3222291575,
2147483648, 1073741824, 2684354560, 1342177280, 2281701376, 1946157056, 436207616, 2566914048, 2625634304, 3208642560, 2720006144, 2098200576, 111673344, 2354315264, 3464626176, 4027383808, 2886631424, 3770826752, 1691164672, 3357462528, 1993345024, 3752330240, 873073152, 2870150400, 1700563072, 87021376, 1097028000, 1222351248, 1560027592, 2977959924, 23268898, 437609937
	};

	struct utility_function {
	private:

		// 格林码 
		__device__ __host__ static uint grayCode(uint i) {
			return i ^ (i >> 1);
		}

		// 生成第 d 维度的第 i 个 sobol 数
		__device__ __host__ static float sobol(uint d, uint i) {
			uint result = 0;
			uint offset = d * 32;
			for (uint j = 0; i; i >>= 1, j++) {
				if (i & 1)
					result ^= soboal_V[j + offset];
			}

			return float(result) * (1.0f / float(0xFFFFFFFFU));
		}

		// 生成第 i 帧的第 b 次反弹需要的二维随机向量




		__device__ __host__ static glm::vec2 CranleyPattersonRotation(glm::vec2 p) {
			uint pseed = uint(
				uint((pix_x[get_thread_idx()] * 0.5 + 0.5) * core_para::IMAGE_WIDTH()) * uint(1973) +
				uint((pix_y[get_thread_idx()] * 0.5 + 0.5) * core_para::IMAGE_HEIGHT()) * uint(9277) +
				uint(114514 / 1919) * uint(26699)) | uint(1);

			float u = float(wang_hash(pseed)) / 4294967296.0;
			float v = float(wang_hash(pseed)) / 4294967296.0;

			p.x += u;
			if (p.x > 1) p.x -= 1;
			if (p.x < 0) p.x += 1;

			p.y += v;
			if (p.y > 1) p.y -= 1;
			if (p.y < 0) p.y += 1;

			return p;
		}


	public:

		__device__ __host__ static uint wang_hash(uint seed) {
			seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
			seed *= uint(9);
			seed = seed ^ (seed >> 4);
			seed *= uint(0x27d4eb2d);
			seed = seed ^ (seed >> 15);
			return seed;
		}

		__device__ __host__ static float rand() {
			uint seed = uint(
				uint((pix_x[get_thread_idx()] * 0.5 + 0.5) * core_para::IMAGE_WIDTH()) * uint(1973) +
				uint((pix_y[get_thread_idx()] * 0.5 + 0.5) * core_para::IMAGE_HEIGHT()) * uint(9277) +
				uint(pixel_sample_cnt[get_thread_idx()]) * uint(26699)) | uint(1);
			return float(wang_hash(seed)) / 4294967296.0;
		}
		__device__ __host__ static unsigned int get_thread_idx() {
			return threadIdx.x;
		}

		__device__ __host__ static glm::vec2 sobolVec2(uint i, uint b) {

			glm::vec2 uv = { sobol(b * 2, grayCode(i)) ,sobol(b * 2 + 1, grayCode(i)) };

			uv = CranleyPattersonRotation(uv);

			return uv;
		}

		__device__ __host__ static glm::vec3 translate_to_normal_hemisphere(const glm::vec3& N, const glm::vec3& v) {
			glm::vec3 helper = glm::vec3(1, 0, 0);
			if (abs(N.x) > 0.999) helper = glm::vec3(0, 0, 1);
			glm::vec3 tangent = glm::normalize(glm::cross(N, helper));
			glm::vec3 bitangent = glm::normalize(glm::cross(N, tangent));
			return v.x * tangent + v.y * bitangent + v.z * N;
		}

		__device__ __host__ static glm::vec3 tone_mapping(const glm::vec3& color, float limit) {
			glm::vec3 hdr;
			float luminance = 0.3 * color.x + 0.6 * color.y + 0.1 * color.z;
			hdr = color * 1.0f / (1.0f + luminance / limit);
			hdr = glm::pow(hdr, glm::vec3(1.0f / 2.2));
			return hdr;
		}


		__device__ __host__ static float GTR2(float NdotH, float a)
		{
			float a2 = a * a;
			float t = 1 + (a2 - 1) * NdotH * NdotH;
			return a2 / (math_const_value::PI() * t * t);
		}

		__device__ __host__ static float SchlickFresnel(float u) {
			float m = glm::clamp<float>(1 - u, 0, 1);
			float m2 = m * m;
			return m2 * m2 * m; // pow(m,5)
		}

		__device__ __host__ static float smithG_GGX(float NdotV, float alphaG) {
			float a = alphaG * alphaG;
			float b = NdotV * NdotV;
			return 1 / (NdotV + sqrt(a + b - a * b));
		}

		__device__ __host__ static int get_global_idx() {
			int nx = blockDim.x * blockDim.y;
			int offset = blockDim.x * threadIdx.y + threadIdx.x;
			int ix = nx * blockIdx.x;
			return ix + offset;
		}

	};

}

