#include <iostream>


// -- CURRENT VERSION --
// -- FOR LOOP IN KERNEL VERSION --
// -- WITH OFFSET --
__global__ void cdist3(const float   * ref,
                        int           ref_nb,
                        const float * query,
                        int           query_nb,
                        int           batches,
                        int           dim,
                        int           paddedDim,
                        int           offset,
                        float       * d_gpu_dist,
                        int         * d_index){

    // we need 3 * paddedDim * sizeof(float) shared memory
    extern __shared__ float smem[];

    // unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int tid = threadIdx.x;

    for(unsigned int query_index=0; query_index<query_nb; query_index++){
        // initialize smem, if tid < dim, copy data, else copy 0
        smem[tid]               = (tid < dim) ? ((ref[(tid*ref_nb)+blockIdx.x]) * (query[(tid*query_nb*batches)+query_index+offset]))      : 0;
        smem[tid+paddedDim]     = (tid < dim) ? ((ref[(tid*ref_nb)+blockIdx.x]) * (ref[(tid*ref_nb)+blockIdx.x]))           : 0;
        smem[tid+(2*paddedDim)] = (tid < dim) ? ((query[(tid*query_nb*batches)+query_index+offset]) * (query[(tid*query_nb*batches)+query_index+offset])) : 0;

        // perform first reduction step when copying data
        if (tid + blockDim.x < dim){
            smem[tid]               += ((ref[((tid+blockDim.x)*ref_nb)+blockIdx.x]) * (query[((tid+blockDim.x)*query_nb*batches)+query_index+offset]));
            smem[tid+paddedDim]     += ((ref[((tid+blockDim.x)*ref_nb)+blockIdx.x]) * (ref[((tid+blockDim.x)*ref_nb)+blockIdx.x]));
            smem[tid+(2*paddedDim)] += ((query[((tid+blockDim.x)*query_nb*batches)+query_index+offset]) * (query[((tid+blockDim.x)*query_nb*batches)+query_index+offset]));
        }

        __syncthreads();

        for(unsigned int s=blockDim.x/2; s>0; s>>=1){
            if(tid < s){
                smem[tid]                += smem[tid + s];
                smem[tid+blockDim.x]     += smem[tid + s + blockDim.x];
                smem[tid+(2*blockDim.x)] += smem[tid + s + (2*blockDim.x)];
            }
            __syncthreads();
        }

        // write result for this block to global memory
        if (tid == 0){
            d_gpu_dist[(query_nb*blockIdx.x)+query_index] = smem[0] / (sqrt(smem[blockDim.x]) * sqrt(smem[2*blockDim.x]));
            d_index[(query_nb*blockIdx.x)+query_index] = blockIdx.x;
        }
    }

}




// -- USED FOR PICK_K ON GPU --
/*
    * Find min using reduction, then set corresponding value to 1 in gpu_dist and repeat k times
    * Handles distances and indexes
    * Handles offset for streams
*/
__global__ void get_min_intrablock3(const float* gpu_dist,
                                    const int  * gpu_index,
                                    int          query_index,
                                    int          offset,
                                    int          query_nb,
                                    float      * min_candidates,
                                    int        * min_index_candidates){

    // set up shared mem
    // 2 * blockDim * sizeof(float) for distances and indexes
    extern __shared__ float smem[];
    
    // copy distances and indexes to shared mem
    smem[threadIdx.x]            =  gpu_dist[(query_nb*blockDim.x*blockIdx.x)+(threadIdx.x*query_nb) + offset + query_index];
    smem[threadIdx.x+blockDim.x] = gpu_index[(query_nb*blockDim.x*blockIdx.x)+(threadIdx.x*query_nb) + offset + query_index];

    __syncthreads();

    // find min
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(threadIdx.x < s){
            if(smem[threadIdx.x] > smem[threadIdx.x + s]){
                smem[threadIdx.x] = smem[threadIdx.x + s];
                smem[threadIdx.x+blockDim.x] = smem[threadIdx.x + s + blockDim.x];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0){
        // if(smem[0]==0) printf("STORED ZERO! smem[%d]: %f\n", blockIdx.x, smem[0]);
              min_candidates[(blockIdx.x*query_nb)+query_index+offset] = smem[0];
        min_index_candidates[(blockIdx.x*query_nb)+query_index+offset] = smem[blockDim.x];
    }  

}

// -- USED FOR PICK_K ON GPU --
/*
    * Find min using reduction, then set corresponding value to 1 in gpu_dist and repeat k times
    * Handles distances and indexes
*/
__global__ void get_min_interblock3(const float* min_candidates,
                                    const int  * min_index_candidates,
                                    float     * gpu_dist,
                                    int        query_index,
                                    int        offset,
                                    int        query_nb,
                                    int         i,              // #ith min
                                    int         k,
                                    float      * knn_dist,
                                    int        * knn_index){

    // set up shared mem
    extern __shared__ float smem[];
    
    // copy distances and indexes to shared mem
    smem[threadIdx.x] = min_candidates[(threadIdx.x*query_nb)+query_index+offset];
    smem[threadIdx.x+blockDim.x] = min_index_candidates[(threadIdx.x*query_nb)+query_index+offset];

    __syncthreads();

    // find min
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(threadIdx.x < s){
            if(smem[threadIdx.x] > smem[threadIdx.x + s]){
                smem[threadIdx.x] = smem[threadIdx.x + s];
                smem[threadIdx.x+blockDim.x] = smem[threadIdx.x + s + blockDim.x];
            }
        }
        __syncthreads();
    }

    // write result for this block to global memory
    if (threadIdx.x == 0){
        int min_index = smem[blockDim.x];

        knn_dist[(query_nb*i)+query_index+offset] = smem[0];
        knn_index[(query_nb*i)+query_index+offset] = min_index;

        // printf("placing %f at index: %d\n", smem[0], (k*query_index)+i);

        // set gpu_dist["min_index"] to 1
        gpu_dist[(min_index*query_nb)+query_index+offset] = 1;

    }
}




__global__ void get_min_interblock5(const float * min_candidates,
                                    const int   * min_index_candidates,
                                    int           size,
                                    float       * gpu_dist,
                                    int           query_index,
                                    int           query_nb,
                                    int           offset,
                                    int           batches,
                                    int           i,              // #ith min
                                    int           k,
                                    float       * knn_dist,
                                    int         * knn_index){

    // set up shared mem
    extern __shared__ float smem[];
    
    // copy distances and indexes to shared mem
    smem[threadIdx.x]            = (threadIdx.x < size) ?       min_candidates[(threadIdx.x*query_nb)+query_index] : 1;
    smem[threadIdx.x+blockDim.x] = (threadIdx.x < size) ? min_index_candidates[(threadIdx.x*query_nb)+query_index] : 1;

    // printf("smem[%d]: %f\n", threadIdx.x, smem[threadIdx.x]);

    // if(smem[threadIdx.x] == 0) printf("LOADED ZERO! smem[%d]: %f\n", threadIdx.x, smem[threadIdx.x]);
    __syncthreads();

    // find min
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(threadIdx.x < s){
            if(smem[threadIdx.x] > smem[threadIdx.x + s]){
                smem[threadIdx.x] = smem[threadIdx.x + s];
                smem[threadIdx.x+blockDim.x] = smem[threadIdx.x + s + blockDim.x];
            }
        }
        __syncthreads();
    }

    // write result for this block to global memory
    if (threadIdx.x == 0){
        int min_index = smem[blockDim.x];

        knn_dist[(query_nb*batches*i)+query_index+offset] = smem[0];
        knn_index[(query_nb*batches*i)+query_index+offset] = min_index;

        // printf("placing %f at index: %d\n", smem[0], (k*query_index)+i);

        // set gpu_dist["min_index"] to 1
        gpu_dist[(min_index*query_nb)+query_index] = 1;

    }
}

// -- USED FOR PICK_K ON GPU --
/*
    * Find min using reduction, then set corresponding value to 1 in gpu_dist and repeat k times
    * Handles distances and indexes
    * Handles offset for streams
*/
__global__ void get_min_intrablock6(const float* gpu_dist,
                                    const int  * gpu_index,
                                    int          offset,
                                    int          query_nb,
                                    int          batch,
                                    int          batches,
                                    float      * min_candidates,
                                    int        * min_index_candidates){

    // set up shared mem
    // 2 * blockDim * sizeof(float) for distances and indexes
    extern __shared__ float smem[];
    
    for(unsigned int query_index=batch*(query_nb/batches); query_index<query_nb/batches; query_index++){
        // copy distances and indexes to shared mem
        smem[threadIdx.x]            =  gpu_dist[(query_nb*blockDim.x*blockIdx.x)+(threadIdx.x*query_nb) + offset + query_index];
        smem[threadIdx.x+blockDim.x] = gpu_index[(query_nb*blockDim.x*blockIdx.x)+(threadIdx.x*query_nb) + offset + query_index];

        __syncthreads();

        // find min
        for(unsigned int s=blockDim.x/2; s>0; s>>=1){
            if(threadIdx.x < s){
                if(smem[threadIdx.x] > smem[threadIdx.x + s]){
                    smem[threadIdx.x] = smem[threadIdx.x + s];
                    smem[threadIdx.x+blockDim.x] = smem[threadIdx.x + s + blockDim.x];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0){
            // if(smem[0]==0) printf("STORED ZERO! smem[%d]: %f\n", blockIdx.x, smem[0]);
                min_candidates[(blockIdx.x*query_nb)+query_index+offset] = smem[0];
            min_index_candidates[(blockIdx.x*query_nb)+query_index+offset] = smem[blockDim.x];
        }
    }

}

__global__ void get_min_interblock6(const float * min_candidates,
                                    const int   * min_index_candidates,
                                    int           size,
                                    float       * gpu_dist,
                                    int           query_nb,
                                    int           offset,
                                    int           batches,
                                    int           i,              // #ith min
                                    int           k,
                                    float       * knn_dist,
                                    int         * knn_index){

    // set up shared mem
    extern __shared__ float smem[];
    
    for(unsigned int query_index=0; query_index<query_nb; query_index++){
        // copy distances and indexes to shared mem
        smem[threadIdx.x]            = (threadIdx.x < size) ?       min_candidates[(threadIdx.x*query_nb)+query_index] : 1;
        smem[threadIdx.x+blockDim.x] = (threadIdx.x < size) ? min_index_candidates[(threadIdx.x*query_nb)+query_index] : 1;

        // printf("smem[%d]: %f\n", threadIdx.x, smem[threadIdx.x]);

        // if(smem[threadIdx.x] == 0) printf("LOADED ZERO! smem[%d]: %f\n", threadIdx.x, smem[threadIdx.x]);
        __syncthreads();

        // find min
        for(unsigned int s=blockDim.x/2; s>0; s>>=1){
            if(threadIdx.x < s){
                if(smem[threadIdx.x] > smem[threadIdx.x + s]){
                    smem[threadIdx.x] = smem[threadIdx.x + s];
                    smem[threadIdx.x+blockDim.x] = smem[threadIdx.x + s + blockDim.x];
                }
            }
            __syncthreads();
        }

        // write result for this block to global memory
        if (threadIdx.x == 0){
            int min_index = smem[blockDim.x];

            knn_dist[(query_nb*batches*i)+query_index+offset] = smem[0];
            knn_index[(query_nb*batches*i)+query_index+offset] = min_index;

            // printf("placing %f at index: %d\n", smem[0], (k*query_index)+i);

            // set gpu_dist["min_index"] to 1
            gpu_dist[(min_index*query_nb)+query_index] = 1;

        }
    }
}



__global__ void gpu_custom_insertion_sort(float *dist, int *index, int length, int k, int query_nb, int batches, int offset, float *knn_dist, int *knn_index){
    
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Initialize the first index
    index[tid] = 0;

    // Go through all points
    for (int i = 1; i < length; ++i) {

        // Store current distance and associated index
        float curr_dist = dist[(i * query_nb) + tid];
        int curr_index = i;

        // Skip the current value if its index is >= k and if it's higher than the k-th already sorted smallest value
        if (i >= k && curr_dist >= dist[(k - 1) * query_nb + tid]) {
            continue;
        }

        // Shift values (and indexes) higher than the current distance to the right
        int j = min(i, k - 1);
        while (j > 0 && dist[(j - 1) * query_nb + tid] > curr_dist) {
            dist[(j * query_nb) + tid] = dist[((j - 1) * query_nb) + tid];
            index[(j * query_nb) + tid] = index[((j - 1) * query_nb) + tid];
            --j;
        }

        // Write the current distance and index at their position
        dist[(j * query_nb) + tid] = curr_dist;
        index[(j * query_nb) + tid] = curr_index;
    }

    // Copy the k smallest distances and associated index to the knn_dist and knn_index arrays
    for (int i = 0; i < k; i++) {
        knn_dist[(query_nb*batches*i)+tid+offset] = dist[(i * query_nb) + tid];
        knn_index[(query_nb*batches*i)+tid+offset] = index[(i * query_nb) + tid];
    }
}

// PROPOSED SOLUTION 1
bool your_solution(const float * ref,
                     int           ref_nb,
                     const float * query,
                     int           query_nb,
                     int           dim,
                     int           k,
                     float *       knn_dist,    // output fields
                     int *         knn_index) {

    // ds that must be allocated
    float   *d_ref, *d_query;

    cudaMalloc(&d_ref, ref_nb * dim * sizeof(float));
    cudaMalloc(&d_query, query_nb * dim * sizeof(float));

    cudaMemcpy(  d_ref,   ref, ref_nb * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, query_nb * dim * sizeof(float), cudaMemcpyHostToDevice);

    float *d_knn_dist;
    int *d_knn_index;

    cudaMalloc(&d_knn_dist, query_nb * k * sizeof(float));
    cudaMalloc(&d_knn_index, query_nb * k * sizeof(int));

    // Get device properties
    int deviceId = 0;
    cudaSetDevice(deviceId);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    uint64_t o_matrix_size = 1L * ref_nb * query_nb * sizeof(float);

    uint64_t batches = ceil((double) 2*o_matrix_size/(deviceProp.totalGlobalMem));

    int gridSize = ref_nb;      // Number of blocks for cdist

    for(unsigned int batch = 0; batch<batches; batch++){
        
        // printf("batch %d\n", batch);

        float   *d_gpu_dist;
        int     *d_index;

        cudaMalloc(&d_gpu_dist, o_matrix_size/batches);
        cudaMalloc(&d_index, o_matrix_size/batches);

        // Calculate the next power of 2 for dim
        int nextPow2 = 1;
        while (nextPow2 < dim) {
            nextPow2 <<= 1;
        }

        // Calculate the number of elements required in smem with padding
        int paddedDim = nextPow2/2;
        int smemSize = 3 * paddedDim * sizeof(float);

        cdist3<<< gridSize, paddedDim, smemSize >>>(d_ref, ref_nb, d_query, query_nb/batches, batches, dim, paddedDim, batch*(query_nb/batches),d_gpu_dist, d_index);

        // batch k selection ----------------------------------------------------------------------------------------------------------------------
        int blockSize2 = 1024;
        int gridSize2 = ref_nb/blockSize2;

        nextPow2 = 1;
        while (nextPow2 < gridSize2) {
            nextPow2 <<= 1;
        }

        int interblockGridSize = nextPow2;

        // allocate cuda mem
        float *d_min_distances;
        int *d_min_indexes;

        cudaMalloc(&d_min_distances, query_nb/batches * gridSize2 * sizeof(float));
        cudaMalloc(&d_min_indexes,   query_nb/batches * gridSize2 * sizeof(int));

        for(unsigned int i=0; i<k; i++){
            for(unsigned int query_index=0; query_index<query_nb/batches; query_index++){
                get_min_intrablock3<<< gridSize2, blockSize2, 2 * blockSize2 * sizeof(float) >>>(d_gpu_dist, d_index, query_index, 0, query_nb/batches, d_min_distances, d_min_indexes);
                get_min_interblock5<<< 1, interblockGridSize, 2 * interblockGridSize * sizeof(float) >>>(d_min_distances, d_min_indexes, gridSize2, d_gpu_dist, query_index,  query_nb/batches, batch*(query_nb/batches),batches, i, k, d_knn_dist, d_knn_index);
            }
        }

        // for(unsigned int i=0; i<k; i++){
        //     get_min_intrablock6<<< gridSize2, blockSize2, 2 * blockSize2 * sizeof(float) >>>(d_gpu_dist, d_index, 0, query_nb/batches, batch, batches, d_min_distances, d_min_indexes);
        //     get_min_interblock6<<< 1, interblockGridSize, 2 * interblockGridSize * sizeof(float) >>>(d_min_distances, d_min_indexes, gridSize2, d_gpu_dist,  query_nb/batches, batch*(query_nb/batches),batches, i, k, d_knn_dist, d_knn_index);
        // }


        cudaFree(d_gpu_dist);
        cudaFree(d_index);
        cudaFree(d_min_distances);
        cudaFree(d_min_indexes);
    }



    // mem copy back to cpu
    cudaMemcpy(knn_dist, d_knn_dist, query_nb * k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(knn_index, d_knn_index, query_nb * k * sizeof(int), cudaMemcpyDeviceToHost);


    cudaFree(d_ref);
    cudaFree(d_query);
    cudaFree(d_knn_dist);
    cudaFree(d_knn_index);

    return true;
}

// PROPOSED SOLUTION 2 (better version)
bool your_solution2(const float * ref,
                     int           ref_nb,
                     const float * query,
                     int           query_nb,
                     int           dim,
                     int           k,
                     float *       knn_dist,    // output fields
                     int *         knn_index) {

    // ds that must be allocated
    float   *d_ref, *d_query;

    cudaMalloc(&d_ref, ref_nb * dim * sizeof(float));
    cudaMalloc(&d_query, query_nb * dim * sizeof(float));

    cudaMemcpy(  d_ref,   ref, ref_nb * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, query_nb * dim * sizeof(float), cudaMemcpyHostToDevice);

    float *d_knn_dist;
    int *d_knn_index;

    cudaMalloc(&d_knn_dist, query_nb * k * sizeof(float));
    cudaMalloc(&d_knn_index, query_nb * k * sizeof(int));

    // Get device properties
    int deviceId = 0;
    cudaSetDevice(deviceId);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    // std::cout << "Total GPU Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB\n";

    uint64_t o_matrix_size = 1L * ref_nb * query_nb * sizeof(float);

    uint64_t batches = ceil((double) 2*o_matrix_size/(deviceProp.totalGlobalMem));
    // std::cout << "batches = " << o_matrix_size << "/" << deviceProp.totalGlobalMem << " = " << batches << "\n";  

    int gridSize = ref_nb;      // Number of blocks for cdist

    for(unsigned int batch = 0; batch<batches; batch++){
        
        // printf("batch %d\n", batch);

        float   *d_gpu_dist;
        int     *d_index;

        cudaMalloc(&d_gpu_dist, o_matrix_size/batches);
        cudaMalloc(&d_index, o_matrix_size/batches);

        // Calculate the next power of 2 for dim
        int nextPow2 = 1;
        while (nextPow2 < dim) {
            nextPow2 <<= 1;
        }

        // Calculate the number of elements required in smem with padding
        int paddedDim = nextPow2/2;
        int smemSize = 3 * paddedDim * sizeof(float);

        cdist3<<< gridSize, paddedDim, smemSize >>>(d_ref, ref_nb, d_query, query_nb/batches, batches, dim, paddedDim, batch*(query_nb/batches), d_gpu_dist, d_index);

        // printf("end cdist\n");

        // batch k selection ----------------------------------------------------------------------------------------------------------------------
        int blockSize2 = 1024;
        int gridSize2 = query_nb/blockSize2;

        gpu_custom_insertion_sort<<< gridSize2, blockSize2>>>(d_gpu_dist, d_index, ref_nb, k, query_nb/batches, batches, 0, d_knn_dist, d_knn_index);

        // printf("end k selection\n");

        cudaFree(d_gpu_dist);
        cudaFree(d_index);
    }



    // mem copy back to cpu
    cudaMemcpy(knn_dist, d_knn_dist, query_nb * k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(knn_index, d_knn_index, query_nb * k * sizeof(int), cudaMemcpyDeviceToHost);


    cudaFree(d_ref);
    cudaFree(d_query);
    cudaFree(d_knn_dist);
    cudaFree(d_knn_index);

    return true;
}

