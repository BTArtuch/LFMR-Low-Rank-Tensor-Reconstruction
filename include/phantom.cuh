#pragma once

#include <cuda_runtime.h>

namespace phantom {

__global__ void makeEllipse(float* image, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    // Map pixel centers to [-1, 1] in each dimension.
    const float px = 2.0f * (x + 0.5f) / width - 1.0f;
    const float py = 2.0f * (y + 0.5f) / height - 1.0f;

    constexpr float rx = 0.65f;
    constexpr float ry = 0.80f;

    const float normalizedRadius =
        (px * px) / (rx * rx) + (py * py) / (ry * ry);

    image[y * width + x] = normalizedRadius <= 1.0f ? 1.0f : 0.0f;
}

inline void launchEllipse(float* deviceImage, int width, int height)
{
    const dim3 block(16, 16);
    const dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    makeEllipse<<<grid, block>>>(deviceImage, width, height);
}

} // namespace phantom