#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// Define a simple kernel that does some computation
__global__ void intensiveKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float f = idx;
    for(int i = 0; i < n; i++) {
        f = sinf(f) * cosf(f) * tanf(f);  // Intensive math operations
        if (idx < n) {
            data[idx] += f;  // Use the allocated memory to prevent it from being optimized away
        }
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

    // Ask the user for the amount of memory to allocate
    float memoryInGB;
    std::cout << "Enter the amount of memory to allocate (in GBs): ";
    std::cin >> memoryInGB;

    // Allocate the specified amount of GPU RAM
    const size_t dataSize = static_cast<size_t>(memoryInGB * 1024 * 1024 * 1024);  // Convert GB to bytes
    float *d_data;
    cudaError_t err = cudaMalloc(&d_data, dataSize);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Time the kernel execution
    auto start = std::chrono::high_resolution_clock::now();

    // Launch the kernel multiple times to ensure it runs for at least 5-10 minutes
    while (true) {
        intensiveKernel<<<1000000, 256>>>(d_data, dataSize / sizeof(float));  // Adjust the loop count inside the kernel if needed
        cudaDeviceSynchronize();

        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(now - start);

        if (elapsed.count() >= 10) {  // Stop after 10 minutes
            break;
        }
    }

    // Free the allocated GPU memory
    cudaFree(d_data);

    std::cout << "Kernel execution complete." << std::endl;

    return 0;
}
