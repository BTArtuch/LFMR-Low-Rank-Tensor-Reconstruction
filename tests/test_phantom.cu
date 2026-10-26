#include "phantom.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

bool checkCuda(cudaError_t error, const char* operation)
{
    if (error == cudaSuccess) {
        return true;
    }

    std::cerr << operation << " failed: "
              << cudaGetErrorString(error) << '\n';
    return false;
}

bool testEllipse(int width, int height)
{
    const size_t count = static_cast<size_t>(width) * height;
    float* deviceImage = nullptr;

    if (!checkCuda(cudaMalloc(&deviceImage, count * sizeof(float)),
                   "cudaMalloc")) {
        return false;
    }

    phantom::launchEllipse(deviceImage, width, height);

    bool passed = checkCuda(cudaGetLastError(), "kernel launch") &&
                  checkCuda(cudaDeviceSynchronize(), "kernel execution");

    std::vector<float> image(count);
    if (passed) {
        passed = checkCuda(
            cudaMemcpy(image.data(), deviceImage, count * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy");
    }

    cudaFree(deviceImage);

    if (!passed) {
        return false;
    }

    // Check every pixel against the expected baseline ellipse.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float px = 2.0f * (x + 0.5f) / width - 1.0f;
            const float py = 2.0f * (y + 0.5f) / height - 1.0f;
            const float radius =
                (px * px) / (0.65f * 0.65f) +
                (py * py) / (0.80f * 0.80f);
            const float expected = radius <= 1.0f ? 1.0f : 0.0f;
            const float actual = image[static_cast<size_t>(y) * width + x];

            if (actual != expected) {
                std::cerr << "Mismatch at (" << x << ", " << y
                          << "): expected " << expected
                          << ", got " << actual << '\n';
                return false;
            }
        }
    }

    // Basic sanity checks: center is inside; a corner is outside.
    const int centerX = width / 2;
    const int centerY = height / 2;
    if (image[static_cast<size_t>(centerY) * width + centerX] != 1.0f) {
        std::cerr << "Ellipse center should be inside.\n";
        return false;
    }

    // For a 1x1 image, its only pixel center is also the image center,
    // which is inside the ellipse. Check an outside corner only when the
    // dimensions are large enough for that pixel center to be outside.
    if (width >= 3 && height >= 3 && image[0] != 0.0f) {
        std::cerr << "Image corner should be outside.\n";
        return false;
    }

    return true;
}

} // namespace

int main()
{
    // Include dimensions not divisible by the 16x16 CUDA block size.
    const int dimensions[][2] = {
        {256, 256},
        {127, 93},
        {1, 1}
    };

    for (const auto& size : dimensions) {
        if (!testEllipse(size[0], size[1])) {
            std::cerr << "Ellipse test failed for "
                      << size[0] << "x" << size[1] << '\n';
            return EXIT_FAILURE;
        }
    }

    std::cout << "All phantom tests passed.\n";
    return EXIT_SUCCESS;
}