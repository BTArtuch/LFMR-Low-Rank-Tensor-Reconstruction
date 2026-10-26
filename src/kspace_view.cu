#include "cufft_roundtrip.cuh"
#include "phantom.cuh"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
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

} // namespace

int main(int argc, char** argv)
{
    const int width = (argc > 1) ? std::atoi(argv[1]) : 256;
    const int height = (argc > 2) ? std::atoi(argv[2]) : 256;
    const char* outputPath = (argc > 3) ? argv[3] : "outputs/kspace.pgm";

    if (width <= 0 || height <= 0) {
        std::cerr << "Width and height must be positive.\n";
        return EXIT_FAILURE;
    }

    const size_t pixelCount = static_cast<size_t>(width) * height;
    const size_t imageBytes = pixelCount * sizeof(float);

    const int packedWidth = width / 2 + 1;
    const size_t spectrumCount = static_cast<size_t>(height) * packedWidth;
    const size_t spectrumBytes = spectrumCount * sizeof(cufftComplex);

    float* deviceImage = nullptr;
    cufftComplex* deviceSpectrum = nullptr;

    if (!checkCuda(cudaMalloc(&deviceImage, imageBytes), "allocate image") ||
        !checkCuda(cudaMalloc(&deviceSpectrum, spectrumBytes),
                   "allocate spectrum")) {
        if (deviceSpectrum) cudaFree(deviceSpectrum);
        if (deviceImage) cudaFree(deviceImage);
        return EXIT_FAILURE;
    }

    phantom::launchEllipse(deviceImage, width, height);

    bool success =
        checkCuda(cudaGetLastError(), "launch phantom kernel") &&
        checkCuda(cudaDeviceSynchronize(), "execute phantom kernel") &&
        checkCufft(fft::forwardReal2D(
                       deviceImage, deviceSpectrum, width, height),
                   "forwardReal2D") &&
        checkCuda(cudaDeviceSynchronize(), "complete forward FFT");

    std::vector<cufftComplex> packedSpectrum(spectrumCount);

    if (success) {
        success = checkCuda(
            cudaMemcpy(packedSpectrum.data(), deviceSpectrum, spectrumBytes,
                       cudaMemcpyDeviceToHost),
            "copy spectrum to host");
    }

    cudaFree(deviceSpectrum);
    cudaFree(deviceImage);

    if (!success) {
        return EXIT_FAILURE;
    }

    // Expand the packed real-input spectrum into a centered full-size image.
    // Apply log compression, then scale the strongest value to white.
    std::vector<float> display(pixelCount);
    float maxLogMagnitude = 0.0f;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Coordinates in the unshifted FFT spectrum.
            const int kx = (x + (width + 1) / 2) % width;
            const int ky = (y + (height + 1) / 2) % height;

            cufftComplex value{};

            if (kx < packedWidth) {
                value = packedSpectrum[
                    static_cast<size_t>(ky) * packedWidth + kx];
            } else {
                // Hermitian symmetry: F(ky,kx) = conj(F(-ky,-kx)).
                const int mirrorY = (height - ky) % height;
                const int mirrorX = width - kx;
                value = packedSpectrum[
                    static_cast<size_t>(mirrorY) * packedWidth + mirrorX];
            }

            const float magnitude = std::hypot(value.x, value.y);
            const float logMagnitude = std::log1p(magnitude);

            display[static_cast<size_t>(y) * width + x] = logMagnitude;
            maxLogMagnitude = std::max(maxLogMagnitude, logMagnitude);
        }
    }

    std::ofstream output(outputPath, std::ios::binary);
    if (!output) {
        std::cerr << "Could not open output file: " << outputPath << '\n';
        return EXIT_FAILURE;
    }

    output << "P5\n" << width << ' ' << height << "\n255\n";

    for (float value : display) {
        const float scaled = maxLogMagnitude > 0.0f
                                 ? value / maxLogMagnitude
                                 : 0.0f;
        const unsigned char pixel =
            static_cast<unsigned char>(std::round(scaled * 255.0f));
        output.write(reinterpret_cast<const char*>(&pixel), 1);
    }

    std::cout << "Wrote centered log-magnitude k-space image to "
              << outputPath << '\n';
    return EXIT_SUCCESS;
}