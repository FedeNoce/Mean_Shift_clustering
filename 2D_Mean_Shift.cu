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


#define N 100000
#define MAX_ITER 20
#define BANDWIDTH 2
#define TPB 128

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
    d_shifted_datapoints_x[idx] = newPosition.x ;
    d_shifted_datapoints_y[idx] = newPosition.y;


}

int main() {
    srand(time(NULL));   // Initialization, should only be called once.
    FILE *fpt;
    const char *file_name = "/home/federico/CLionProjects/Mean_Shift_clustering/datasets/2D_data_100000.csv";
    fpt = fopen(file_name, "r");
    printf("%s\n", file_name);

    //allocate memory on the device for the data points
    float *d_original_datapoints_x;
    float *d_original_datapoints_y;
    float *d_shifted_datapoints_x;
    float *d_shifted_datapoints_y;



    cudaMalloc(&d_original_datapoints_x, N*sizeof(float));
    cudaMalloc(&d_original_datapoints_y, N*sizeof(float));
    cudaMalloc(&d_shifted_datapoints_x, N*sizeof(float));
    cudaMalloc(&d_shifted_datapoints_y, N*sizeof(float));


    //allocate memory for host
    float *h_original_datapoints_x = (float *) malloc(N*sizeof(float));
    float *h_original_datapoints_y = (float *) malloc(N*sizeof(float));
    float *h_shifted_datapoints_x = (float *) malloc(N*sizeof(float));
    float *h_shifted_datapoints_y = (float *) malloc(N*sizeof(float));



    //initalize datapoints from csv
    //printf("DataPoints: \n");
    for (int i = 0; i < N; ++i) {
        fscanf(fpt, "%f,%f\n", &h_original_datapoints_x[i], &h_original_datapoints_y[i]);
        //printf("(%f, %f) \n",  h_original_datapoints_x[i], h_original_datapoints_y[i]);
        h_shifted_datapoints_x[i] = h_original_datapoints_x[i];
        h_shifted_datapoints_y[i] = h_original_datapoints_y[i];

    }
    fclose(fpt);




    //copy datapoints and all other data from host to device
    cudaMemcpy(d_original_datapoints_x, h_original_datapoints_x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_original_datapoints_y, h_original_datapoints_y, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shifted_datapoints_x, h_shifted_datapoints_x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shifted_datapoints_y, h_shifted_datapoints_y, N*sizeof(float), cudaMemcpyHostToDevice);


    //Start time for clustering
    clock_t start = clock();
    int cur_iter = 0;

    while (cur_iter < MAX_ITER) {
        MeanShift<<<(N+TPB - 1) / TPB, TPB>>>(d_original_datapoints_x, d_original_datapoints_y, d_shifted_datapoints_x, d_shifted_datapoints_y);

        cur_iter++;
    }
    clock_t end = clock();
    float seconds = (float) (end - start) / CLOCKS_PER_SEC;
    printf("Time for clustering: %f s \n", seconds);
    printf("Time for average iteration: %f s\n", seconds / MAX_ITER);
    cudaMemcpy(h_shifted_datapoints_x, d_shifted_datapoints_x, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_shifted_datapoints_y, d_shifted_datapoints_y, N*sizeof(float), cudaMemcpyDeviceToHost);


    for (int i = 0; i < N; ++i) {
        if (h_shifted_datapoints_x[i] != 0.0) {
            float current_centroid_x = h_shifted_datapoints_x[i];

            for (int j = i + 1; j < N; j++) {
                if (abs(h_shifted_datapoints_x[j] - current_centroid_x) < EPSILON) {
                    h_shifted_datapoints_x[j] = 0.0;
                }
            }
        }
    }
    printf("Centroids: \n");
    for (int i = 0; i < N; ++i) {
        if (h_shifted_datapoints_x[i] != 0.0) {
            printf("(%f, %f) \n", h_shifted_datapoints_x[i], h_shifted_datapoints_y[i]);
        }
    }


    free(h_shifted_datapoints_x);
    free(h_shifted_datapoints_y);
    free(h_original_datapoints_x);
    free(h_original_datapoints_y);

    cudaFree(d_shifted_datapoints_x);
    cudaFree(d_shifted_datapoints_y);
    cudaFree(d_original_datapoints_x);
    cudaFree(d_original_datapoints_y);


    return 0;
}
