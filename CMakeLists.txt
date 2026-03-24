cmake_minimum_required(VERSION 4.1)
project(CudaIntro CUDA CXX)

set(CMAKE_CUDA_STANDARD 23)


add_executable(CudaIntro vecAdd.cu)

add_executable(MatMultiplication matMul.cu)

add_executable(rawMatMul matMulRaw.cpp)

set_target_properties(rawMatMul PROPERTIES
        CXX_STANDARD 23)

set_target_properties(CudaIntro PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
