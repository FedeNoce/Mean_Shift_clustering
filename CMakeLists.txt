cmake_minimum_required(VERSION 3.20)
project(Mean_Shift_clustering CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Mean_Shift_clustering 2D_Mean_Shift.cu)

set_target_properties(Mean_Shift_clustering PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
