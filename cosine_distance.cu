#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <assert.h>


/*
 * Compute the cosine distance between two vectors
 * inspired from Cuda webinar on reduction kernel03 (mabye extend optimization to kernel04)
*/

__global__ void gpu_get_components(const float * ref,
                                int           ref_nb,
                                const float * query,
                                int           query_nb,
                                int           dim,
                                int           ref_index,
                                int           query_index,
                                float       * d_odot,
                                float       * d_odenom_a,
                                float       * d_odenom_b,
                                float       * d_gpu_dist){
    
    extern __shared__ float smem[];

    // each thread loads one element from global to shared mem (but in this case we need product of corresponding dimension)
    unsigned int tid = threadIdx.x;
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    // dot product
    smem[tid]                = (ref[(i*ref_nb)+ref_index]) * (query[(i*query_nb)+query_index]);
    // denom_a
    smem[tid+blockDim.x]     = (ref[(i*ref_nb)+ref_index]) * (ref[(i*ref_nb)+ref_index]);
    // denom_b
    smem[tid+(2*blockDim.x)] = (query[(i*query_nb)+query_index]) * (query[(i*query_nb)+query_index]);

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
    // THIS PART IS NOT NECESSARY IF WORKING WITH ONLY ONE BLOCK
    // if (tid == 0){
    //     d_odot[blockIdx.x]    = smem[0];
    //     d_odenom_a[blockIdx.x] = smem[blockDim.x];
    //     d_odenom_b[blockIdx.x] = smem[2*blockDim.x];
    // } 

    if (tid == 0){
        d_gpu_dist[(query_nb*ref_index)+query_index] = smem[0] / (sqrt(smem[blockDim.x]) * sqrt(smem[2*blockDim.x]));
    }

}

// get idot,idenom_a and idenom_b. Respectively sum them up, and then calculate final result which will be written into gpu_dist
// THIS FUNCTION IS NOT NECESSARY IF WORKING WITH ONLY ONE BLOCK
__global__ void gpu_cosine_distance(int     ref_nb,
                                    int     query_nb,
                                    int     ref_index,
                                    int     query_index,
                                    float * d_idot,
                                    float * d_idenom_a,
                                    float * d_idenom_b,
                                    float * d_gpu_dist){

    extern __shared__ float smem[];

    unsigned int tid = threadIdx.x;
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    smem[tid] = d_idot[i];
    smem[tid+blockDim.x] = d_idenom_a[i];
    smem[tid+(2*blockDim.x)] = d_idenom_b[i];

    __syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            smem[tid] += smem[tid + s];
            smem[tid+blockDim.x] += smem[tid + s + blockDim.x];
            smem[tid+(2*blockDim.x)] += smem[tid + s + (2*blockDim.x)];
        }
        __syncthreads();
    }

    // write cosine distance result in global memory
    if (tid == 0){
        d_gpu_dist[(query_nb*ref_index)+query_index] = smem[0] / (sqrt(smem[blockDim.x]) * sqrt(smem[2*blockDim.x]));
    }

}

__global__ void cdist(const float * ref,
                        int           ref_nb,
                        const float * query,
                        int           query_nb,
                        int           dim,
                        int           query_index,
                        float       * d_gpu_dist){

    // we need 3 * blockDim * sizeof(float) shared memory
    extern __shared__ float smem[];

    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int tid = threadIdx.x;

    // printf("i: %d\n", i);

    // dot product
    smem[tid]                = (ref[(tid*ref_nb)+blockIdx.x]) * (query[(tid*query_nb)+query_index]);
    // denom_a
    smem[tid+blockDim.x]     = (ref[(tid*ref_nb)+blockIdx.x]) * (ref[(tid*ref_nb)+blockIdx.x]);
    // denom_b
    smem[tid+(2*blockDim.x)] = (query[(tid*query_nb)+query_index]) * (query[(tid*query_nb)+query_index]);



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


void initialize_data(float * ref,
                     int     ref_nb,
                     float * query,
                     int     query_nb,
                     int     dim) {

    // Initialize random number generator
    srand(42); // to get the same results

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



int main(void) {

    
    
    // Parameters 0 (to develop your solution)
    // const int ref_nb   = 4096;
    // const int query_nb = 1024;
    // const int dim      = 64;
    // const int k        = 16;

    // Parameters 1
    const int ref_nb   = 16384;
    const int query_nb = 4096;
    const int dim      = 128;
    const int k        = 100;

    // Parameters 2
    // const int ref_nb   = 163840;
    // const int query_nb = 40960;
    // const int dim      = 128;
    // const int k        = 16;

    // Parameters 3
    // const int ref_nb   = 16384;
    // const int query_nb = 4096;
    // const int dim      = 1280;
    // const int k        = 16;

    // Parameters 4
    // const int ref_nb   = 5;
    // const int query_nb = 3;
    // const int dim      = 128;
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

    float * cpu_dist   = (float*) malloc(ref_nb * query_nb * sizeof(float));
    float * h_gpu_dist   = (float*) malloc(ref_nb * query_nb * sizeof(float));

    // Allocation checks
    if (!ref || !query || !cpu_dist || !h_gpu_dist) {
        printf("Error: Memory allocation error\n"); 
        free(ref);
	    free(query);
	    free(cpu_dist);
	    free(h_gpu_dist);
        return EXIT_FAILURE;
    }

    // Initialize reference and query points with random values
    initialize_data(ref, ref_nb, query, query_nb, dim);

    printf("Performing cosine distance computation on CPU\n");

    // start timer
    struct timeval  tv1_cpu, tv2_cpu;
    gettimeofday(&tv1_cpu, NULL);

    // Perform cosine distance computation on CPU
    for(int i=0; i<ref_nb; ++i) {
        for(int j=0; j<query_nb; ++j) {
            cpu_dist[i*query_nb+j] = cosine_distance(ref, ref_nb, query, query_nb, dim, i, j);
        }
    }

    // stop timer
    gettimeofday(&tv2_cpu, NULL);

    // compute and print the elapsed time in millisec
    printf ("Total time = %f milliseconds\n",
             (double) (1000.0 * (tv2_cpu.tv_sec - tv1_cpu.tv_sec) + (tv2_cpu.tv_usec - tv1_cpu.tv_usec) / 1000.0));

    // print results
    // for(int i=0; i<ref_nb; ++i) {
    //     for(int j=0; j<query_nb; ++j) {
    //         printf("%f ||", cpu_dist[i*query_nb+j]);
    //     }
    //     printf("\n");
    // }
    
    printf("Performing cosine distance computation on GPU\n");



    printf("blockSize: %d\n", blockSize);
    printf("gridSize: %d\n", gridSize);

    // copy ref and query into cuda mem
    float *d_ref, *d_query;
    float *d_odot, *d_odenom_a, *d_odenom_b;
    float *d_gpu_dist;


    cudaMalloc(&d_ref, ref_nb * dim * sizeof(float));
    cudaMalloc(&d_query, ref_nb * dim * sizeof(float));

    cudaMalloc(&d_odot, gridSize * sizeof(float));
    cudaMalloc(&d_odenom_a, gridSize * sizeof(float));
    cudaMalloc(&d_odenom_b, gridSize * sizeof(float));

    cudaMalloc(&d_gpu_dist, ref_nb * query_nb * sizeof(float));

    // printf("Copying data from host to device\n");
    cudaMemcpy(  d_ref,   ref, ref_nb * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, ref_nb * dim * sizeof(float), cudaMemcpyHostToDevice);
    // printf("Done\n\n");

    // start timer
    struct timeval  tv1, tv2;
    gettimeofday(&tv1, NULL);

    // int num_stream = ref_nb;
    // cudaStream_t *streams = (cudaStream_t *) malloc(num_stream * sizeof(cudaStream_t));

    // for(int i=0; i<num_stream; i++){
    //     cudaStreamCreate(&streams[i]);
    // }

    // for(int i=0; i<query_nb; i++){
    //     for(int j=0; j<ref_nb; j++){
    //         gpu_get_components<<< gridSize, blockSize, 3 * blockSize * sizeof(float), streams[j] >>>(d_ref, ref_nb, d_query, query_nb, dim, j, i, d_odot, d_odenom_a, d_odenom_b);            
    //         gpu_cosine_distance<<< 1, gridSize, 3 * gridSize * sizeof(float), streams[j] >>>(ref_nb, query_nb, j, i, d_odot, d_odenom_a, d_odenom_b, d_gpu_dist);
    //     }
    // }


    // for(unsigned int i=0; i<query_nb; i++){
    //     // printf("query %d\n", i);
    //     for(unsigned int j=0; j<ref_nb; j++){
    //         gpu_get_components<<< gridSize, blockSize, 3 * blockSize * sizeof(float) >>>(d_ref, ref_nb, d_query, query_nb, dim, j, i, d_odot, d_odenom_a, d_odenom_b, d_gpu_dist);            
    //         // gpu_cosine_distance<<< 1, gridSize, 3 * gridSize * sizeof(float) >>>(ref_nb, query_nb, j, i, d_odot, d_odenom_a, d_odenom_b, d_gpu_dist);
    //     }
    // }

    for(unsigned int query_index=0; query_index<query_nb; query_index++){
        // printf("Query %d\n", query_index);
        cdist<<< gridSize, blockSize, 3 * blockSize * sizeof(float) >>>(d_ref, ref_nb, d_query, query_nb, dim, query_index, d_gpu_dist);
    }


    // stop timer
    gettimeofday(&tv2, NULL);

    // compute and print the elapsed time in millisec
    printf ("Total time = %f milliseconds\n",
             (double) (1000.0 * (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) / 1000.0));


    //mem copy back to cpu
    // expensive
    cudaMemcpy(h_gpu_dist, d_gpu_dist, ref_nb * query_nb * sizeof(float), cudaMemcpyDeviceToHost);

    // print results
            // for(int i=0; i<ref_nb; ++i) {
            //     for(int j=0; j<query_nb; ++j) {
            //         // printf("%f ||",fabs(h_gpu_dist[i*query_nb+j] - cpu_dist[i*query_nb+j]));
            //         printf("%f ||", h_gpu_dist[i*query_nb+j]);
            //     }
            //     printf("\n");
            // }

    // check results
    int error = 0;
    for(int i=0; i<ref_nb; ++i) {
        for(int j=0; j<query_nb; ++j) {
            if(fabs(h_gpu_dist[i*query_nb+j] - cpu_dist[i*query_nb+j]) > 0.001){
                // printf("Error at index (%d, %d)\t", i, j);
                // printf("CPU: %f || GPU: %f\n", cpu_dist[i*query_nb+j], h_gpu_dist[i*query_nb+j]);
                error++;
            }
        }
    }

    printf("Number of errors: %d\n", error);
    printf("Percentage of errors: %f\n", (float) error / (ref_nb * query_nb) * 100);



    // free cuda mem
    cudaFree(d_ref);
    cudaFree(d_query);
    cudaFree(d_odot);
    cudaFree(d_odenom_a);
    cudaFree(d_odenom_b);
    cudaFree(d_gpu_dist);

}