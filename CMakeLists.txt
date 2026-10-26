cmake_minimum_required(VERSION 3.18)
project(mri_tensor_recon LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 50)

find_package(CUDAToolkit REQUIRED)

add_executable(phantom
    src/phantom.cu
)

target_include_directories(phantom PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

set_target_properties(phantom PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

enable_testing()

function(add_cuda_test target_name)
    add_executable(${target_name} ${ARGN})

    target_include_directories(${target_name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )

    add_test(
        NAME ${target_name}
        COMMAND ${target_name}
    )
endfunction()

add_cuda_test(test_phantom
    tests/test_phantom.cu
)

add_cuda_test(test_cufft_roundtrip
    tests/test_cufft_roundtrip.cu
    src/cufft_roundtrip.cu
)

target_link_libraries(test_cufft_roundtrip PRIVATE
    CUDA::cufft
)

add_executable(kspace_view
    src/kspace_view.cu
    src/cufft_roundtrip.cu
)

target_include_directories(kspace_view PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(kspace_view PRIVATE
    CUDA::cufft
)

set_target_properties(kspace_view PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)