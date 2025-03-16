#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<utility.h>
namespace LuTracer {
    cudaError check_cf_(cudaError t, const char* filename, int line) {
        if (t != cudaSuccess) {
            printf("cuda error:\n      code:%d\n      name:%s\n      information:%s\n      file:%s, line:%d\n", t, cudaGetErrorName(t), cudaGetErrorString(t), filename, line);
        }
        return t;
    }

    using uint = unsigned int;


}

