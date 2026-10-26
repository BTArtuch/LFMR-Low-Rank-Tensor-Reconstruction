#include "cufft_roundtrip.cuh"

namespace fft {

cufftResult forwardReal2D(
    const float* deviceImage,
    cufftComplex* deviceSpectrum,
    int width,
    int height)
{
    if (deviceImage == nullptr || deviceSpectrum == nullptr) {
        return CUFFT_INVALID_VALUE;
    }

    if (width <= 0 || height <= 0) {
        return CUFFT_INVALID_SIZE;
    }

    cufftHandle plan;
    cufftResult result = cufftPlan2d(&plan, height, width, CUFFT_R2C);
    if (result != CUFFT_SUCCESS) {
        return result;
    }

    // cuFFT's C API takes a non-const input pointer, although this
    // out-of-place forward transform does not modify the input image.
    result = cufftExecR2C(
        plan,
        const_cast<cufftReal*>(deviceImage),
        deviceSpectrum);

    const cufftResult destroyResult = cufftDestroy(plan);
    return result != CUFFT_SUCCESS ? result : destroyResult;
}

cufftResult inverseReal2D(
    cufftComplex* deviceSpectrum,
    float* deviceImage,
    int width,
    int height)
{
    if (deviceSpectrum == nullptr || deviceImage == nullptr) {
        return CUFFT_INVALID_VALUE;
    }

    if (width <= 0 || height <= 0) {
        return CUFFT_INVALID_SIZE;
    }

    cufftHandle plan;
    cufftResult result = cufftPlan2d(&plan, height, width, CUFFT_C2R);
    if (result != CUFFT_SUCCESS) {
        return result;
    }

    result = cufftExecC2R(plan, deviceSpectrum, deviceImage);

    const cufftResult destroyResult = cufftDestroy(plan);
    return result != CUFFT_SUCCESS ? result : destroyResult;
}

} // namespace fft