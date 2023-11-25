#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <assert.h>
#include <inttypes.h>

// 2023-03-07 16:00:00 NVPROF
    //         Type  Time(%)      Time     Calls       Avg       Min       Max  Name
    //   API calls:   95.81%  1.33104s         3  443.68ms  538.90us  1.32840s  cudaMalloc
    //                 2.67%  37.120ms      1070  34.691us  4.7000us  1.5229ms  cudaLaunchKernel
    //                 0.89%  12.330ms         2  6.1648ms  1.9446ms  10.385ms  cudaMemcpy
    //                 0.63%  8.7707ms         1  8.7707ms  8.7707ms  8.7707ms  cuDeviceGetPCIBusId

// -- OLD VERSION --
/*
    * Compute the cosine distance between two vectors
    * inspired from Cuda webinar on reduction kernel03 (mabye extend optimization to kernel04)
    * In this version, each block computes a single cosine distance between a variable ref point vs a query point, each block has "dim" threads
*/
__global__ void cdist(const float   * ref,
                        int           ref_nb,
                        const float * query,
                        int           query_nb,
                        int           dim,
                        int           query_index,
                        float       * d_gpu_dist){

    // we need 3 * blockDim * sizeof(float) shared memory
    extern __shared__ float smem[];

    // unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int tid = threadIdx.x;

    // dot product
    smem[tid]                = ((ref[(tid*ref_nb)+blockIdx.x]) * (query[(tid*query_nb)+query_index])) ;
    // denom_a
    smem[tid+blockDim.x]     = (ref[(tid*ref_nb)+blockIdx.x]) * (ref[(tid*ref_nb)+blockIdx.x]) ;
    // denom_b
    smem[tid+(2*blockDim.x)] = (query[(tid*query_nb)+query_index]) * (query[(tid*query_nb)+query_index]) ;

    if(smem[tid] == 0){
        printf("smem[%d]: %f\n", tid, smem[tid]);
        printf("smem[%d]: %f\n", tid+blockDim.x, smem[tid+blockDim.x]);
        printf("smem[%d]: %f\n", tid+(2*blockDim.x), smem[tid+(2*blockDim.x)]);
    }


    __syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            smem[tid] += smem[tid + s];
            smem[tid+blockDim.x] += smem[tid + s + blockDim.x];
            smem[tid+(2*blockDim.x)] += smem[tid + s + (2*blockDim.x)];
        }
        __syncthreads();
    }

    // write result for this block to global memory
    if (tid == 0){
        d_gpu_dist[(query_nb*blockIdx.x)+query_index] = smem[0] / (sqrt(smem[blockDim.x]) * sqrt(smem[2*blockDim.x]));
    }
}


// -- OLD VERSION --
/*
    * Compute the cosine distance between two vectors
    * inspired from Cuda webinar on reduction kernel04
    * Half the number of threads per block
*/
__global__ void cdist2(const float   * ref,
                        int           ref_nb,
                        const float * query,
                        int           query_nb,
                        int           dim,
                        int           query_index,
                        float       * d_gpu_dist){

    // we need 3 * blockDim * sizeof(float) shared memory
    extern __shared__ float smem[];

    // unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int tid = threadIdx.x;

    // dot product
    smem[tid]                = ((ref[(tid*ref_nb)+blockIdx.x]) * (query[(tid*query_nb)+query_index])) +((ref[((tid+blockDim.x)*ref_nb)+blockIdx.x]) * (query[((tid+blockDim.x)*query_nb)+query_index]));
    // denom_a
    smem[tid+blockDim.x]     = ((ref[(tid*ref_nb)+blockIdx.x]) * (ref[(tid*ref_nb)+blockIdx.x])) + ((ref[((tid+blockDim.x)*ref_nb)+blockIdx.x]) * (ref[((tid+blockDim.x)*ref_nb)+blockIdx.x]));
    // denom_b
    smem[tid+(2*blockDim.x)] = ((query[(tid*query_nb)+query_index]) * (query[(tid*query_nb)+query_index])) + ((query[((tid+blockDim.x)*query_nb)+query_index]) * (query[((tid+blockDim.x)*query_nb)+query_index]));


    __syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            smem[tid] += smem[tid + s];
            smem[tid+blockDim.x] += smem[tid + s + blockDim.x];
            smem[tid+(2*blockDim.x)] += smem[tid + s + (2*blockDim.x)];
        }
        __syncthreads();
    }

    // write result for this block to global memory
    if (tid == 0){
        d_gpu_dist[(query_nb*blockIdx.x)+query_index] = smem[0] / (sqrt(smem[blockDim.x]) * sqrt(smem[2*blockDim.x]));
    }
}


// -- CURRENT VERSION --
/*
    * Compute the cosine distance between two vectors
    * inspired from Cuda webinar on reduction kernel04
    * Half the number of threads per block
    * Use padding to handle non Po2 dimensions
    * This version can handle more than 1280 dimensions
*/
__global__ void padded_cdist(const float   * ref,
                        int           ref_nb,
                        const float * query,
                        int           query_nb,
                        int           dim,
                        int           paddedDim,
                        int           query_index,
                        float       * d_gpu_dist){

    // we need 3 * paddedDim * sizeof(float) shared memory
    extern __shared__ float smem[];

    // unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int tid = threadIdx.x;

    smem[tid] = (tid < dim) ? ((ref[(tid*ref_nb)+blockIdx.x]) * (query[(tid*query_nb)+query_index])) : 0;
    smem[tid+paddedDim] = (tid < dim) ? ((ref[(tid*ref_nb)+blockIdx.x]) * (ref[(tid*ref_nb)+blockIdx.x])) : 0;
    smem[tid+(2*paddedDim)] = (tid < dim) ? ((query[(tid*query_nb)+query_index]) * (query[(tid*query_nb)+query_index])) : 0;

    if (tid + blockDim.x < dim){
        smem[tid] += ((ref[((tid+blockDim.x)*ref_nb)+blockIdx.x]) * (query[((tid+blockDim.x)*query_nb)+query_index]));
        smem[tid+paddedDim] += ((ref[((tid+blockDim.x)*ref_nb)+blockIdx.x]) * (ref[((tid+blockDim.x)*ref_nb)+blockIdx.x]));
        smem[tid+(2*paddedDim)] += ((query[((tid+blockDim.x)*query_nb)+query_index]) * (query[((tid+blockDim.x)*query_nb)+query_index]));
    }

    __syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            smem[tid] += smem[tid + s];
            smem[tid+blockDim.x] += smem[tid + s + blockDim.x];
            smem[tid+(2*blockDim.x)] += smem[tid + s + (2*blockDim.x)];
        }
        __syncthreads();
    }

    // write result for this block to global memory
    if (tid == 0){
        d_gpu_dist[(query_nb*blockIdx.x)+query_index] = smem[0] / (sqrt(smem[blockDim.x]) * sqrt(smem[2*blockDim.x]));
    }
}


// naive get nearest neighbors
// repeat k times:
//      find minimum at position pos
//      add that pos to knn array
//      set that position to infinity

__global__ void get_min_intrablock(const float* gpu_dist,
                                    int          query_index,
                                    int          query_nb,
                                    float      * min_candidates){

    // set up shared mem
    // blockDim * sizeof(float) for distances and indexes
    extern __shared__ float smem[];
    
    // copy distances and indexes to shared mem
    smem[threadIdx.x] = gpu_dist[(query_nb*blockIdx.x)+threadIdx.x];



    if(smem[threadIdx.x] == 0){
        printf("d smem[%d]: %f\n", threadIdx.x, smem[threadIdx.x]);
        printf("d gpu_dist[%d]: %f\n", (query_nb*blockIdx.x)+threadIdx.x, gpu_dist[(query_nb*blockIdx.x)+threadIdx.x]);
    }



    // printf("smem[%d]: %f\n", threadIdx.x, smem[threadIdx.x]);

    __syncthreads();

    // find min
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(threadIdx.x < s){
            if(smem[threadIdx.x] > smem[threadIdx.x + s]){
                smem[threadIdx.x] = smem[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    // write result for this block to global memory
    if (threadIdx.x == 0){
        // printf("min: %f\n", smem[0]);
        min_candidates[(query_nb*blockIdx.x)+query_index] = smem[0];
    }

}


__global__ void get_min_interblock(const float* min_candidates,
                                    float      * min_dist){

    // set up shared mem
    extern __shared__ float smem[];
    
    // copy distances and indexes to shared mem
    smem[threadIdx.x] = min_candidates[threadIdx.x];

    __syncthreads();

    // find min
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(threadIdx.x < s){
            if(smem[threadIdx.x] > smem[threadIdx.x + s]){
                smem[threadIdx.x] = smem[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    // write result for this block to global memory
    if (threadIdx.x == 0){
        *min_dist = smem[0];
    }

}

// Initialize data randomly
void initialize_data(float * ref,
                     int     ref_nb,
                     float * query,
                     int     query_nb,
                     int     dim,
                     int     seed = 51) {

    // Initialize random number generator
    srand(seed); // to get the same results

    // Generate random reference points
    for (int i=0; i<ref_nb*dim; ++i) {
        ref[i] = 10. * (float)(rand() / (double)RAND_MAX);
        // ref[i] = 1;
    }

    // Generate random query points
    for (int i=0; i<query_nb*dim; ++i) {
        query[i] = 10. * (float)(rand() / (double)RAND_MAX);
        // query[i] = 1;
    }
}

// CPU implementation of cosine distance computation
float cosine_distance(const float * ref,
                       int           ref_nb,
                       const float * query,
                       int           query_nb,
                       int           dim,
                       int           ref_index,
                       int           query_index) {

   double dot = 0.0, denom_a = 0.0, denom_b = 0.0 ;
     for(unsigned int d = 0u; d < dim; ++d) {
        dot += ref[d * ref_nb + ref_index] * query[d * query_nb + query_index] ;
        denom_a += ref[d * ref_nb + ref_index] * ref[d * ref_nb + ref_index] ;
        denom_b += query[d * query_nb + query_index] * query[d * query_nb + query_index] ;
    }
    
    return dot / (sqrt(denom_a) * sqrt(denom_b)) ;
    // return dot;
    // return denom_a;
    // return denom_b;
}


void print_matrix(float * matrix, int rows, int cols){
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            printf("%f ||", matrix[i*cols+j]);
        }
        printf("\n");
    }
}


int main(void) {
    
    // Parameters 0 (to develop your solution)
    const int ref_nb   = 4096;
    const int query_nb = 1024;
    const int dim      = 1280;
    const int k        = 16;

    // Parameters 1
    // const int ref_nb   = 16384;
    // const int query_nb = 4096;
    // const int dim      = 128;
    // const int k        = 100;

    // Parameters 2     (not working: too many query & ref points)
    // const int ref_nb   = 163840;
    // const int query_nb = 40960;
    // const int dim      = 128;
    // const int k        = 16;

    // Parameters 3     (not working: too many dimensions)
    // const int ref_nb   = 16384;
    // const int query_nb = 4096;
    // const int dim      = 1280;
    // const int k        = 16;

    // Parameters 4
    // const int ref_nb   = 5;
    // const int query_nb = 3;
    // const int dim      = 1280;
    // const int k        = 4;


    int blockSize = dim;        // Number of threads per block (this approach cannot handle more than 1024 threads) (last case scenario)
    int gridSize = ref_nb;      // Number of blocks
    

    // Display
    printf("PARAMETERS\n");
    printf("- Number reference points : %d\n",   ref_nb);
    printf("- Number query points     : %d\n",   query_nb);
    printf("- Dimension of points     : %d\n",   dim);
    printf("- Number of neighbors     : %d\n\n", k);


    // Sanity check
    if (ref_nb<k) {
        printf("Error: k value is larger that the number of reference points\n");
        return EXIT_FAILURE;
    }

    // Allocate input points and output k-NN distances / indexes
    float * ref        = (float*) malloc(ref_nb   * dim * sizeof(float));
    float * query      = (float*) malloc(query_nb * dim * sizeof(float));

    uint64_t o_matrix_size = 1LL * ref_nb * query_nb * sizeof(float);

    float * cpu_dist   = (float*) malloc(o_matrix_size);
    float * h_gpu_dist   = (float*) malloc(o_matrix_size);

    float * knn_dist   = (float*) malloc(o_matrix_size);
    int   * knn_index  = (int*)   malloc(o_matrix_size);

    

    // Allocation checks
    if (!ref || !query || !cpu_dist || !h_gpu_dist || !knn_dist || !knn_index) {
        printf("Error: Memory allocation error\n"); 
        free(ref);
	    free(query);
	    free(cpu_dist);
	    free(h_gpu_dist);
        free(knn_dist);
        free(knn_index);
        return EXIT_FAILURE;
    }

    // Initialize reference and query points with random values
    initialize_data(ref, ref_nb, query, query_nb, dim);

    printf("Performing cosine distance computation on CPU\n");

    // start timer
    struct timeval  tv1_cpu, tv2_cpu;
    gettimeofday(&tv1_cpu, NULL);

    // Perform cosine distance computation on CPU
    for(unsigned int query_index=0; query_index<query_nb; query_index++){
        if(query_index % 100 == 0) printf("Query %d\n", query_index);
        for(unsigned int ref_index=0; ref_index<ref_nb; ref_index++){
            cpu_dist[(query_nb*ref_index)+query_index] = cosine_distance(ref, ref_nb, query, query_nb, dim, ref_index, query_index);
        }
    }

    // stop timer
    gettimeofday(&tv2_cpu, NULL);

    // compute and print the elapsed time in millisec
    printf ("Total time = %f milliseconds\n",
             (double) (1000.0 * (tv2_cpu.tv_sec - tv1_cpu.tv_sec) + (tv2_cpu.tv_usec - tv1_cpu.tv_usec) / 1000.0));

    // print results
    // print_matrix(cpu_dist, ref_nb, query_nb);
    
    printf("Performing cosine distance computation on GPU\n");

    printf("blockSize: %d\n", blockSize);
    printf("gridSize: %d\n", gridSize);

    // copy ref and query into cuda mem
    float *d_ref, *d_query;
    float *d_gpu_dist;

    float *d_min_distances;                                                 // min dist for each block
    float *d_min_dist;                                                      // min dist for each query point

    cudaMalloc(&d_ref, ref_nb * dim * sizeof(float));
    cudaMalloc(&d_query, ref_nb * dim * sizeof(float));

    cudaMalloc(&d_gpu_dist, o_matrix_size);
    cudaMemset(d_gpu_dist, 100, o_matrix_size);

    cudaMemcpy(  d_ref,   ref, ref_nb * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, ref_nb * dim * sizeof(float), cudaMemcpyHostToDevice);

    


    // start timer 
    struct timeval  tv1, tv2;
    gettimeofday(&tv1, NULL);

    // Calculate the next power of 2 for dim
    int nextPow2 = 1;
    while (nextPow2 < dim) {
        nextPow2 <<= 1;
    }

    // Calculate the number of elements required in smem with padding
    int paddedDim = nextPow2/2;
    int smemSize = 3 * paddedDim * sizeof(float);
    printf("paddedDim: %d\n", paddedDim);

    // test cd2
    for(unsigned int query_index=0; query_index<query_nb; query_index++){
        // printf("Query %d\n", query_index);
        padded_cdist<<< gridSize, paddedDim, smemSize >>>(d_ref, ref_nb, d_query, query_nb, dim, paddedDim, query_index, d_gpu_dist);
    }

    // stop timer
    gettimeofday(&tv2, NULL);


    // compute and print the elapsed time in millisec
    printf ("Total time = %f milliseconds\n",
             (double) (1000.0 * (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) / 1000.0));


    //mem copy back to cpu
    cudaMemcpy(h_gpu_dist, d_gpu_dist, o_matrix_size, cudaMemcpyDeviceToHost);


    // check results
    int error = 0;
    int flag = 0;
    for(int i=0; i<ref_nb; ++i) {
        
        for(int j=0; j<query_nb; ++j) {
            if(fabs(h_gpu_dist[i*query_nb+j] - cpu_dist[i*query_nb+j]) > 0.001){
                // if (1){
                //     printf("Error at index (%d, %d)\n", i, j);
                //     printf("CPU: %f || GPU: %f\n", cpu_dist[i*query_nb+j], h_gpu_dist[i*query_nb+j]);
                // }

                flag = 1;
                
                error++;
            }
        }
    }

     // print results
    // for(int i=0; i<ref_nb; ++i) {
    //     for(int j=0; j<query_nb; ++j) {
    //         printf("%f ||",fabs(h_gpu_dist[i*query_nb+j] - cpu_dist[i*query_nb+j]));
    //         // printf("%f ||", h_gpu_dist[i*query_nb+j]);
    //     }
    //     printf("\n");
    // }


    printf("Number of errors: %d\n", error);
    printf("Percentage of errors: %f\n", (float) error / (1LL * ref_nb * query_nb) * 100);

    


    // -------------- PART 2: k selection -------------------------------------------------------------------------------------------------------------------------------------------------------

    // int blockSize2 = 1024;        // Test to find the best value
    // int gridSize2 = ref_nb/blockSize;

    // cudaMalloc(&d_min_distances, query_nb * gridSize2 * sizeof(float));
    // cudaMalloc(&d_min_dist, query_nb * sizeof(float));

    // float *min_dist;
    // float *min_distances;

    // min_distances = (float*) malloc(query_nb * gridSize2 * sizeof(float));
    

    // printf("\n\nSearching for min\n");
    // printf("blockSize: %d\n", blockSize2);
    // printf("gridSize: %d\n", gridSize2);
    // // select k nearest neighbors
    // for(unsigned int query_index=0; query_index<query_nb; query_index++){
    //     if(query_index == 1){
    //         get_min_intrablock<<< gridSize2, blockSize2, blockSize2 * sizeof(float) >>>(d_gpu_dist, query_index, query_nb, d_min_distances);
    //         printf("intra block min done\n");

    //         // copy min_distances to cpu
    //         cudaMemcpy(min_distances, d_min_distances, query_nb * gridSize2 * sizeof(float), cudaMemcpyDeviceToHost);
            
    //         for(unsigned int i=0; i<gridSize2; i++){
    //             printf("%f ||", min_distances[i]);
    //         }

    //         get_min_interblock<<< 1, gridSize2>>>(d_min_distances, d_min_dist);

    //         // copy min_dist to cpu
    //         cudaMemcpy(min_dist, d_min_dist, sizeof(float), cudaMemcpyDeviceToHost);
    //         printf("min_dist: %f\n", *min_dist);
    //     }
    // }


    // free cuda mem
    cudaFree(d_ref);
    cudaFree(d_query);
    cudaFree(d_gpu_dist);
}