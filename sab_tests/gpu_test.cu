#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// Define a simple kernel that does some computation
__global__ void intensiveKernel(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float f = idx;
    for(int i = 0; i < n; i++) {
        f = sinf(f) * cosf(f) * tanf(f);  // Intensive math operations
    }
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No GPUs found." << std::endl;
        return 1;
    }

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem << " bytes" << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
    }

    // Set the device to the first GPU
    cudaSetDevice(0);

    // Time the kernel execution
    auto start = std::chrono::high_resolution_clock::now();

    // Launch the kernel multiple times to ensure it runs for at least 5-10 minutes
    while (true) {
        intensiveKernel<<<1000000, 256>>>(10000);  // Adjust the loop count inside the kernel if needed
        cudaDeviceSynchronize();

        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(now - start);

        if (elapsed.count() >= 10) {  // Stop after 10 minutes
            break;
        }
    }

    std::cout << "Kernel execution complete." << std::endl;

    return 0;
}
