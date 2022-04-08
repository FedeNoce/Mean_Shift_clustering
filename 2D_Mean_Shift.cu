/*
 ============================================================================
 Name        : 2D_Mean_shift.cu
 Author      : Federico Nocentini & Corso Vignoli
 Version     :
 Copyright   :
 Description : CUDA implementation of Mean Shift clustering algorithm
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>


#define N 10000
#define MAX_ITER 10
#define BANDWIDTH 2
#define TPB 64
#define LAMBDA 1

#define EPSILON 0.1


__global__ void MeanShift(const float *d_original_datapoints_x, const float *d_original_datapoints_y, float *d_shifted_datapoints_x, float *d_shifted_datapoints_y)
{
    //get idx for this datapoint
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    float tot_weight = 0.0;
    float2 newPosition = make_float2(0.0, 0.0);

    if (idx >= N) return;

    float2 shiftedPoint = make_float2(d_shifted_datapoints_x[idx], d_shifted_datapoints_y[idx]);

    for(int i=0; i<N; i++){
        float2 originalPoint = make_float2(d_original_datapoints_x[i], d_original_datapoints_y[i]);
        float2 difference = make_float2(0.0, 0.0);
        difference.x = shiftedPoint.x - originalPoint.x;
        difference.y = shiftedPoint.y - originalPoint.y;
        float squaredDistance = pow(difference.x,2) + pow(difference.y,2);
        float weight= exp(((-squaredDistance)/(2* pow(BANDWIDTH,2))));
        newPosition.x += originalPoint.x * weight;
        newPosition.y += originalPoint.y * weight;
        tot_weight += weight;
    }
    newPosition.x /= tot_weight;
    newPosition.y /= tot_weight;
    d_shifted_datapoints_x[idx] = newPosition.x;
    d_shifted_datapoints_y[idx] = newPosition.y;


}

__device__ float distance_2D(const float x1, const float x2, const float y1, const float y2){
    return sqrt(pow((x1-y1),2) + pow((x2-y2),2));
}

__global__ void Flat_MeanShift(const float *d_original_datapoints_x, const float *d_original_datapoints_y, float *d_shifted_datapoints_x, float *d_shifted_datapoints_y)
{
    //get idx for this datapoint
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    float tot_weight = 0.0;
    float2 newPosition = make_float2(0.0, 0.0);

    if (idx >= N) return;

    float2 shiftedPoint = make_float2(d_shifted_datapoints_x[idx], d_shifted_datapoints_y[idx]);

    for(int i=0; i<N; i++){
        float2 originalPoint = make_float2(d_original_datapoints_x[i], d_original_datapoints_y[i]);
        //int weight = 0;
        if(distance_2D(originalPoint.x, originalPoint.y , shiftedPoint.x, shiftedPoint.y) < LAMBDA){
            //weight = 1;
            newPosition.x += originalPoint.x;
            newPosition.y += originalPoint.y;
            tot_weight += 1;
        }

    }
    newPosition.x /= tot_weight;
    newPosition.y /= tot_weight;
    d_shifted_datapoints_x[idx] = newPosition.x;
    d_shifted_datapoints_y[idx] = newPosition.y;


}


__global__ void getCentroids(float *d_shifted_datapoints_x, float *d_shifted_datapoints_y)
{
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= N) return;

    if (d_shifted_datapoints_x[idx] != 0.0) {
        for (int j = idx + 1; j < N; j++) {
            if (abs(d_shifted_datapoints_x[j] - d_shifted_datapoints_x[idx]) < EPSILON) {
                d_shifted_datapoints_x[j] = 0.0;
                d_shifted_datapoints_y[j] = 0.0;
            }
        }
    }

}

int main() {
    srand(time(NULL));   // Initialization, should only be called once.
    FILE *fpt;
    const char *file_name = "/home/federico/CLionProjects/Mean_Shift_clustering/datasets/2D_data_10000.csv";
    fpt = fopen(file_name, "r");
    printf("%s\n", file_name);

    //allocate memory on the device for the data points
    float *d_original_datapoints_x;
    float *d_original_datapoints_y;
    float *d_shifted_datapoints_x;
    float *d_shifted_datapoints_y;



    //allocate memory on the device for the data points
    cudaMalloc(&d_original_datapoints_x, N*sizeof(float));
    cudaMalloc(&d_original_datapoints_y, N*sizeof(float));
    cudaMalloc(&d_shifted_datapoints_x, N*sizeof(float));
    cudaMalloc(&d_shifted_datapoints_y, N*sizeof(float));

    //allocate memory on the host for the data points
    float *h_original_datapoints_x;
    float *h_original_datapoints_y;
    float *h_shifted_datapoints_x;
    float *h_shifted_datapoints_y;

    //allocate memory on the host for the data points
    cudaMallocHost(&h_original_datapoints_x, N*sizeof(float));
    cudaMallocHost(&h_original_datapoints_y, N*sizeof(float));
    cudaMallocHost(&h_shifted_datapoints_x, N*sizeof(float));
    cudaMallocHost(&h_shifted_datapoints_y, N*sizeof(float));



    //initalize datapoints from csv
    //printf("DataPoints: \n");
    for (int i = 0; i < N; ++i) {
        fscanf(fpt, "%f,%f\n", &h_original_datapoints_x[i], &h_original_datapoints_y[i]);
        h_shifted_datapoints_x[i] = h_original_datapoints_x[i];
        h_shifted_datapoints_y[i] = h_original_datapoints_y[i];

    }
    fclose(fpt);


    //copy datapoints and all other data from host to device
    cudaMemcpy(d_original_datapoints_x, h_original_datapoints_x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_original_datapoints_y, h_original_datapoints_y, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shifted_datapoints_x, h_shifted_datapoints_x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shifted_datapoints_y, h_shifted_datapoints_y, N*sizeof(float), cudaMemcpyHostToDevice);



    for (int i = 0; i < N; ++i) {
        h_shifted_datapoints_x[i]=0.0;
        h_shifted_datapoints_y[i]=0.0;
    }

    //Start time for clustering
    clock_t start = clock();
    int cur_iter = 0;


    while (cur_iter < MAX_ITER) {
        //Choose the kernel
        MeanShift<<<(N+TPB - 1) / TPB, TPB>>>(d_original_datapoints_x, d_original_datapoints_y, d_shifted_datapoints_x, d_shifted_datapoints_y);
        //Flat_MeanShift<<<(N+TPB - 1) / TPB, TPB>>>(d_original_datapoints_x, d_original_datapoints_y, d_shifted_datapoints_x, d_shifted_datapoints_y);
        cur_iter++;
    }

    getCentroids<<<(N+TPB - 1) / TPB, TPB>>>(d_shifted_datapoints_x, d_shifted_datapoints_y);
    cudaMemcpy(h_shifted_datapoints_x, d_shifted_datapoints_x, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_shifted_datapoints_y, d_shifted_datapoints_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    FILE *res;
    res = fopen("/home/federico/CLionProjects/Mean_Shift_clustering/results/2D_data_3_results.csv", "w+");
    printf("Centroids: \n");
    for (int i = 0; i < N; ++i) {
        if (h_shifted_datapoints_x[i] != 0.0) {
            printf("(%f, %f) \n", h_shifted_datapoints_x[i], h_shifted_datapoints_y[i]);
            fprintf(res,"%f, %f\n", h_shifted_datapoints_x[i], h_shifted_datapoints_y[i]);
        }
    }
    clock_t end = clock();
    float seconds = (float) (end - start) / CLOCKS_PER_SEC;
    printf("Time for clustering: %f s \n", seconds);
    printf("Time for average iteration: %f s\n", seconds / MAX_ITER);


    cudaFreeHost(h_shifted_datapoints_x);
    cudaFreeHost(h_shifted_datapoints_y);
    cudaFreeHost(h_original_datapoints_x);
    cudaFreeHost(h_original_datapoints_y);


    cudaFree(d_shifted_datapoints_x);
    cudaFree(d_shifted_datapoints_y);
    cudaFree(d_original_datapoints_x);
    cudaFree(d_original_datapoints_y);


    return 0;
}
