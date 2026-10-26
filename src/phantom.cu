#include "phantom.cuh"

#include <cuda_runtime.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t error = (call);                                          \
        if (error != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(error)         \
                      << " at " << __FILE__ << ":" << __LINE__ << '\n';      \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

int main(int argc, char** argv)
{
    const int width = (argc > 1) ? std::atoi(argv[1]) : 256;
    const int height = (argc > 2) ? std::atoi(argv[2]) : 256;
    const char* outputPath = (argc > 3) ? argv[3] : "phantom.pgm";

    if (width <= 0 || height <= 0) {
        std::cerr << "Width and height must be positive.\n";
        return EXIT_FAILURE;
    }

    const size_t count = static_cast<size_t>(width) * height;
    const size_t bytes = count * sizeof(float);

    float* deviceImage = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceImage, bytes));

    phantom::launchEllipse(deviceImage, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> image(count);
    CUDA_CHECK(cudaMemcpy(
        image.data(), deviceImage, bytes, cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaFree(deviceImage));

    std::ofstream output(outputPath, std::ios::binary);
    if (!output) {
        std::cerr << "Could not open output file: " << outputPath << '\n';
        return EXIT_FAILURE;
    }

    output << "P5\n" << width << ' ' << height << "\n255\n";
    for (float value : image) {
        const unsigned char pixel =
            static_cast<unsigned char>(value * 255.0f);
        output.write(reinterpret_cast<const char*>(&pixel), 1);
    }

    std::cout << "Wrote " << width << "x" << height
              << " phantom to " << outputPath << '\n';

    return EXIT_SUCCESS;
}