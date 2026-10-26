#include "cufft_roundtrip.cuh"
#include "phantom.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

bool checkCuda(cudaError_t result, const char* operation)
{
    if (result == cudaSuccess) {
        return true;
    }

    std::cerr << operation << " failed: "
              << cudaGetErrorString(result) << '\n';
    return false;
}

bool checkCufft(cufftResult result, const char* operation)
{
    if (result == CUFFT_SUCCESS) {
        return true;
    }

    std::cerr << operation << " failed with cuFFT status "
              << static_cast<int>(result) << '\n';
    return false;
}

bool testRoundTrip(int width, int height)
{
    const size_t pixelCount = static_cast<size_t>(width) * height;
    const size_t imageBytes = pixelCount * sizeof(float);

    // For a 2D real-to-complex transform, cuFFT stores only
    // width / 2 + 1 complex values per row.
    const size_t spectrumCount =
        static_cast<size_t>(height) * (width / 2 + 1);
    const size_t spectrumBytes = spectrumCount * sizeof(cufftComplex);

    float* deviceImage = nullptr;
    cufftComplex* deviceSpectrum = nullptr;
    float* deviceReconstruction = nullptr;

    auto cleanup = [&]() {
        if (deviceReconstruction != nullptr) {
            cudaFree(deviceReconstruction);
        }
        if (deviceSpectrum != nullptr) {
            cudaFree(deviceSpectrum);
        }
        if (deviceImage != nullptr) {
            cudaFree(deviceImage);
        }
    };

    if (!checkCuda(cudaMalloc(&deviceImage, imageBytes), "allocate image") ||
        !checkCuda(cudaMalloc(&deviceSpectrum, spectrumBytes),
                   "allocate spectrum") ||
        !checkCuda(cudaMalloc(&deviceReconstruction, imageBytes),
                   "allocate reconstruction")) {
        cleanup();
        return false;
    }

    phantom::launchEllipse(deviceImage, width, height);

    if (!checkCuda(cudaGetLastError(), "launch phantom kernel") ||
        !checkCuda(cudaDeviceSynchronize(), "execute phantom kernel")) {
        cleanup();
        return false;
    }

    // Keep a reference to the exact GPU-generated input.
    std::vector<float> original(pixelCount);
    if (!checkCuda(cudaMemcpy(original.data(), deviceImage, imageBytes,
                              cudaMemcpyDeviceToHost),
                   "copy original image to host")) {
        cleanup();
        return false;
    }

    if (!checkCufft(fft::forwardReal2D(
                        deviceImage, deviceSpectrum, width, height),
                    "forwardReal2D")) {
        cleanup();
        return false;
    }

    if (!checkCufft(fft::inverseReal2D(
                        deviceSpectrum, deviceReconstruction, width, height),
                    "inverseReal2D")) {
        cleanup();
        return false;
    }

    if (!checkCuda(cudaDeviceSynchronize(), "complete FFT round trip")) {
        cleanup();
        return false;
    }

    std::vector<float> reconstructed(pixelCount);
    if (!checkCuda(cudaMemcpy(reconstructed.data(), deviceReconstruction,
                              imageBytes, cudaMemcpyDeviceToHost),
                   "copy reconstruction to host")) {
        cleanup();
        return false;
    }

    cleanup();

    const float scale = 1.0f / static_cast<float>(pixelCount);
    constexpr float tolerance = 2.0e-5f;
    float maxError = 0.0f;

    for (size_t i = 0; i < pixelCount; ++i) {
        const float actual = reconstructed[i] * scale;
        const float error = std::fabs(actual - original[i]);

        if (!std::isfinite(actual)) {
            std::cerr << "Non-finite reconstruction value at index "
                      << i << '\n';
            return false;
        }

        maxError = std::max(maxError, error);

        if (error > tolerance) {
            std::cerr << "Round-trip mismatch for " << width << 'x' << height
                      << " at index " << i
                      << ": expected " << original[i]
                      << ", got " << actual
                      << ", error " << error << '\n';
            return false;
        }
    }

    std::cout << "cuFFT round trip passed for "
              << width << 'x' << height
              << "; maximum absolute error: " << maxError << '\n';

    return true;
}

} // namespace

int main()
{
    // Non-square and odd dimensions also exercise cuFFT's real-spectrum
    // storage layout. The 1x1 case checks the smallest supported image.
    constexpr int dimensions[][2] = {
        {256, 256},
        {127, 93},
        {1, 1}
    };

    for (const auto& size : dimensions) {
        if (!testRoundTrip(size[0], size[1])) {
            return EXIT_FAILURE;
        }
    }

    std::cout << "All cuFFT round-trip tests passed.\n";
    return EXIT_SUCCESS;
}