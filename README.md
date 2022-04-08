# Mean Shift Implementation

![alt text](https://github.com/FedeNoce/Mean_Shift_clustering/blob/master/.idea/p.png)

## About The Project
We implemented the 2D Mean Shift algorithm, with 2 different Kernel, in three different ways:
1) A sequential mode in C++
2) A Parallel mode in OpenMP
3) A Parallel mode in Cuda

### Built With

* [C++](https://isocpp.org/)
* [OpenMP](https://www.openmp.org/)
* [CUDA](https://developer.nvidia.com/cuda-zone)

## Getting Started

In order to get a local copy and run some tests, follow these simple steps.

1. Clone the repo
```sh
git clone https://github.com/FedeNoce/Mean_Shift_clustering.git
```
2. Chose the implementation:  ```MeanShift.cpp``` for sequential, ```MeanShiftOpenMP.cpp``` for parallel with OpenMP, ```2D_Mean_Shift.cu``` for parallel with CUDA.
3. Choose the Kernel to use: ```Gaussian``` or ```Flat```
4. Choose the dataset and copy the file path in the code
5. Set the parameters with your settings
6. Run the tests
7. Evaluate the clustering of the tests running ```evaluate_mean_shift.py``` 
## Authors

* [**Corso Vignoli**](https://github.com/CVignoli)
* [**Federico Nocentini**](https://github.com/FedeNoce)


## Acknowledgments
Parallel Computing © Course held by Professor [Marco Bertini](https://www.unifi.it/p-doc2-2020-0-A-2b333d2d3529-1.html) - Computer Engineering Master Degree @[University of Florence](https://www.unifi.it/changelang-eng.html)
