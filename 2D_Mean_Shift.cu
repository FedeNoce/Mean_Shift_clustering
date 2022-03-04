/*
 ============================================================================
 Name        : 2D_Mean_shift.cu
 Author      : Federico Nocentini
 Version     :
 Copyright   :
 Description : CUDA implementation of K-means clustering algorithm
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>


#define N 100000
#define MAX_ITER 10
#define BANDWIDTH 2
#define TPB 64
#define TILE_WIDTH 64
#define EPSILON 0.1


__global__ void Tiling_MeanShift(const float *d_original_datapoints_x, const float *d_original_datapoints_y, float *d_shifted_datapoints_x, float *d_shifted_datapoints_y)
{
    __shared__ float tile[TILE_WIDTH][2];
    int tx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tx;

    if (idx >= N) return;

    float2 newPosition = make_float2(0.0, 0.0);
    float tot_weight = 0.0;

    for (int tile_i = 0; tile_i < (N - 1) / TILE_WIDTH + 1; ++tile_i) {

        int tile_idx = tile_i * TILE_WIDTH + tx;

        if (tile_idx < N) {
            tile[tx][0] = d_original_datapoints_x[tile_idx];
            tile[tx][1] = d_original_datapoints_y[tile_idx];

        } else {
            tile[tx][0] = 0.0;
            tile[tx][1] = 0.0;
        }
        __syncthreads();

        if(idx < N){
            float x = d_shifted_datapoints_x[idx];
            float y = d_shifted_datapoints_y[idx];
            float2 shiftedPoint = make_float2(x, y);

            for(int i = 0; i < TILE_WIDTH; i++){
                if (tile[i][0] != 0.0 && tile[i][1] != 0.0) {
                    float2 originalPoint = make_float2(tile[i][0], tile[i][1]);
                    float2 difference = make_float2(0.0, 0.0);
                    difference.x = shiftedPoint.x - originalPoint.x;
                    difference.y = shiftedPoint.y - originalPoint.y;
                    float squaredDistance = pow(difference.x,2) + pow(difference.y,2);
                    if(sqrt(squaredDistance) <= BANDWIDTH){
                        float weight= exp(((-squaredDistance)/(2* pow(BANDWIDTH,2))));
                        newPosition.x += originalPoint.x * weight;
                        newPosition.y += originalPoint.y * weight;
                        tot_weight += weight;
                    }
                }
            }
        }
        __syncthreads();
    }
    if(idx < N){
        newPosition.x /= tot_weight;
        newPosition.y /= tot_weight;
        d_shifted_datapoints_x[idx] = newPosition.x ;
        d_shifted_datapoints_y[idx] = newPosition.y;
    }
}

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

int main()
{
    srand(time(NULL));   // Initialization, should only be called once.
    FILE *fpt;
    //FILE *fpt_centroids;

    fpt = fopen("/home/federico/CLionProjects/Mean_Shift_clustering/datasets/2D_data_3.csv", "r");
    //fpt = fopen("/home/federico/CLionProjects/kmeans_cuda/datasets/2D_data_uniform.csv", "r");
    //fpt_centroids = fopen("/home/federico/CLionProjects/kmeans_cuda/datasets/2D_data_3_centroids.csv", "r");

    //allocate memory on the device for the data points
    float *d_original_datapoints_x;
    float *d_original_datapoints_y;
    float *d_shifted_datapoints_x;
    float *d_shifted_datapoints_y;
    //allocate memory on the device for the cluster assignments
    int *d_clust_assn;


    cudaMalloc(&d_original_datapoints_x, N*sizeof(float));
    cudaMalloc(&d_original_datapoints_y, N*sizeof(float));
    cudaMalloc(&d_shifted_datapoints_x, N*sizeof(float));
    cudaMalloc(&d_shifted_datapoints_y, N*sizeof(float));
    cudaMalloc(&d_clust_assn,N*sizeof(int));


    //allocate memory for host
    float *h_original_datapoints_x = (float*)malloc(N*sizeof(float));
    float *h_original_datapoints_y = (float*)malloc(N*sizeof(float));
    float *h_shifted_datapoints_x = (float*)malloc(N*sizeof(float));
    float *h_shifted_datapoints_y = (float*)malloc(N*sizeof(float));
    int *h_clust_assn = (int*)malloc(N*sizeof(int));



    //initalize datapoints from csv
    printf("DataPoints: \n");
    for(int i=0;i<N;++i){
        fscanf(fpt,"%f,%f\n", &h_original_datapoints_x[i], &h_original_datapoints_y[i]);
        printf("(%f, %f) \n",  h_original_datapoints_x[i], h_original_datapoints_y[i]);
        h_shifted_datapoints_x[i] = h_original_datapoints_x[i];
        h_shifted_datapoints_y[i] = h_original_datapoints_y[i];

    }
    fclose(fpt);




    //copy datapoints and all other data from host to device
    cudaMemcpy(d_original_datapoints_x,h_original_datapoints_x,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_original_datapoints_y,h_original_datapoints_y,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_shifted_datapoints_x,h_shifted_datapoints_x,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_shifted_datapoints_y,h_shifted_datapoints_y,N*sizeof(float),cudaMemcpyHostToDevice);


    //Start time for clustering
    clock_t start = clock();
    int cur_iter = 0;

    while(cur_iter < MAX_ITER)
    {
        //Start time for iteration
        clock_t start_iter = clock();
        MeanShift<<<(N+TPB-1)/TPB, TPB>>>(d_original_datapoints_x, d_original_datapoints_y, d_shifted_datapoints_x, d_shifted_datapoints_y);
        clock_t end_iter = clock();
        float seconds = (float)(end_iter - start_iter) / CLOCKS_PER_SEC;
        printf("Iter %d -> Time: %f\n",cur_iter, seconds);
        cur_iter++;
    }
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Time for clustering: %f\n", seconds);
    cudaMemcpy(h_shifted_datapoints_x,d_shifted_datapoints_x,N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_shifted_datapoints_y,d_shifted_datapoints_y,N*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<N;++i){
        printf("(%f, %f) \n", h_shifted_datapoints_x[i], h_shifted_datapoints_y[i]);
    }


/*    for(int i=0;i<N;++i){
        if(h_shifted_datapoints_x[i] != 0.0){
            float current_centroid_x=h_shifted_datapoints_x[i];
            //float current_centroid_y=h_shifted_datapoints_y[i];
            for(int j=i+1;j<N;j++){
                if(abs(h_shifted_datapoints_x[j] - current_centroid_x) < EPSILON){
                    h_shifted_datapoints_x[j]=0.0;
                    //h_shifted_datapoints_y[j]=0.0;
                }
            }
        }
    }
    printf("Centroids: \n");
    for(int i=0;i<N;++i){
        if(h_shifted_datapoints_x[i] != 0.0) {
            printf("(%f, %f) \n", h_shifted_datapoints_x[i], h_shifted_datapoints_y[i]);
        }
    }*/


    FILE *res;

    res = fopen("/home/federico/CLionProjects/Mean_Shift_clustering/results/2D_data_3_results.csv", "w+");
    for(int i=0;i<N;i++){
        fprintf(res,"%d\n", h_clust_assn[i]);
    }


    return 0;
}
