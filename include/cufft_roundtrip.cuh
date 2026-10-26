#pragma once

#include <cufft.h>

namespace fft {

// Compute a 2D real-to-complex forward FFT.
//
// deviceImage:    width * height real values on the GPU.
// deviceSpectrum: height * (width / 2 + 1) cufftComplex values on the GPU.
//
// Returns a cuFFT status code.
cufftResult forwardReal2D(
    const float* deviceImage,
    cufftComplex* deviceSpectrum,
    int width,
    int height);

// Compute a 2D complex-to-real inverse FFT.
//
// deviceSpectrum: height * (width / 2 + 1) cufftComplex values on the GPU.
//                  cuFFT may modify this buffer during the inverse transform.
// deviceImage:    width * height real values on the GPU.
//
// cuFFT does not normalize the inverse. Divide deviceImage values by
// width * height to obtain the normalized inverse transform.
//
// Returns a cuFFT status code.
cufftResult inverseReal2D(
    cufftComplex* deviceSpectrum,
    float* deviceImage,
    int width,
    int height);

} // namespace fft