cmake_minimum_required(VERSION 3.20)
project(Mean_Shift_clustering CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

# added -fopenmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

set(SOURCE_FILES MeanShiftOpenMp.cpp Point.h)

add_executable(Mean_Shift_clustering_cuda 2D_Mean_Shift.cu)

set_target_properties(Mean_Shift_clustering_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(Mean_Shift_clustering_OpenMp MeanShiftOpenMp.cpp Point.h)

add_executable(Mean_Shift_clustering MeanShift.cpp Point.h)


