#define CUTOFF_VALUE 0.75


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


// -- CURRENT VERSION FOR STREAM COMPACTION --
/*
    * Compute the cosine distance between two vectors
    * inspired from Cuda webinar on reduction kernel04
    * Half the number of threads per block compared to previous version
    * Use padding to handle non Po2 dimensions
    * This version can handle more than 1280 dimensions (max 2048)
    * This version is used for stream compaction, it outputs a array of valid distances and indexes
*/
__global__ void padded_cdist_with_valid(const float   * ref,
                        int           ref_nb,
                        const float * query,
                        int           query_nb,
                        int           dim,
                        int           paddedDim,
                        int           query_index,
                        float       * d_gpu_dist,
                        int         * d_index,
                        int         * d_valid){

    // we need 3 * paddedDim * sizeof(float) shared memory
    extern __shared__ float smem[];

    // unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int tid = threadIdx.x;

    // initialize smem, if tid < dim, copy data, else copy 0
    smem[tid]               = (tid < dim) ? ((ref[(tid*ref_nb)+blockIdx.x]) * (query[(tid*query_nb)+query_index]))      : 0;
    smem[tid+paddedDim]     = (tid < dim) ? ((ref[(tid*ref_nb)+blockIdx.x]) * (ref[(tid*ref_nb)+blockIdx.x]))           : 0;
    smem[tid+(2*paddedDim)] = (tid < dim) ? ((query[(tid*query_nb)+query_index]) * (query[(tid*query_nb)+query_index])) : 0;

    // perform first reduction step when copying data
    if (tid + blockDim.x < dim){
        smem[tid]               += ((ref[((tid+blockDim.x)*ref_nb)+blockIdx.x]) * (query[((tid+blockDim.x)*query_nb)+query_index]));
        smem[tid+paddedDim]     += ((ref[((tid+blockDim.x)*ref_nb)+blockIdx.x]) * (ref[((tid+blockDim.x)*ref_nb)+blockIdx.x]));
        smem[tid+(2*paddedDim)] += ((query[((tid+blockDim.x)*query_nb)+query_index]) * (query[((tid+blockDim.x)*query_nb)+query_index]));
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
        float dist = smem[0] / (sqrt(smem[blockDim.x]) * sqrt(smem[2*blockDim.x]));
        d_gpu_dist[(query_nb*blockIdx.x)+query_index] = dist;
        d_index[(query_nb*blockIdx.x)+query_index] = blockIdx.x;

        // new part for stream compaction
        if(dist < CUTOFF_VALUE){
            d_valid[(query_nb*blockIdx.x)+query_index] = 1;
        }
    }
}

// NOT WORKING +  MUST BE ABLE TO HANDLE ALL DISTS FROM A QUERY
// inclusive prefix sum
// output is a matrix of size query_nb * ref_nb
__global__ void prefix_sum(const int   *valid,
                            int         query_nb,
                            int         query_index,
                            int        *prefix_sum){

    // set up shared mem
    // blockDim * sizeof(float) for distances and indexes
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    
    // copy distances and indexes to shared mem
    sdata[tid] = valid[(query_nb*blockIdx.x)+query_index];

    __syncthreads();

    // prefix sum
    for(unsigned int s=1; s<blockDim.x; s<<=1){
        if(tid >= s){
            sdata[tid] += sdata[tid - s];
        }
        __syncthreads();
    }

    // write result for this block to global memory
    printf("block %d, thread %d, value %d\n", blockIdx.x, tid, sdata[tid]);
    prefix_sum[(query_nb*blockIdx.x)+query_index] = sdata[tid];

}


// get min, add delta check how many candidates are in the range
// TODO: do first reduction step when copying data
__global__ void get_min_intrablock(const float* gpu_dist,
                                    int          query_index,
                                    int          query_nb,
                                    float      * min_candidates){

    // set up shared mem
    // blockDim * sizeof(float) for distances and indexes
    extern __shared__ float smem[];
    
    // copy distances and indexes to shared mem
    smem[threadIdx.x] = gpu_dist[(query_nb*blockDim.x*blockIdx.x)+(threadIdx.x*query_nb)+query_index];

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
        min_candidates[blockIdx.x] = smem[0];

        // // print min candidates
        // printf("Block %d\n", blockIdx.x);
        // for(unsigned int i=0; i<blockDim.x; i++){
        //     printf("%f ||", min_candidates[(query_nb*blockIdx.x)+i]);
        // }
        // printf("\n");
    }

}


__global__ void get_min_interblock(const float* min_candidates,
                                    int         query_nb,
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
        min_dist[query_nb] = smem[0];
    }
}

// -- USED FOR PICK_K ON GPU --
/*
    * Find min using reduction, then set corresponding value to 1 in gpu_dist and repeat k times
    * Handles distances and indexes
*/
__global__ void get_min_intrablock2(const float* gpu_dist,
                                    const int  * gpu_index,
                                    int          query_index,
                                    int          query_nb,
                                    float      * min_candidates,
                                    int        * min_index_candidates){

    // set up shared mem
    // 2 * blockDim * sizeof(float) for distances and indexes
    extern __shared__ float smem[];
    
    // copy distances and indexes to shared mem
    smem[threadIdx.x] = gpu_dist[(query_nb*blockDim.x*blockIdx.x)+(threadIdx.x*query_nb)+query_index];
    smem[threadIdx.x+blockDim.x] = gpu_index[(query_nb*blockDim.x*blockIdx.x)+(threadIdx.x*query_nb)+query_index];

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
        min_candidates[blockIdx.x] = smem[0];
        min_index_candidates[blockIdx.x] = smem[blockDim.x];
    }  

}

// -- USED FOR PICK_K ON GPU --
/*
    * Find min using reduction, then set corresponding value to 1 in gpu_dist and repeat k times
    * Handles distances and indexes
*/
__global__ void get_min_interblock2(const float* min_candidates,
                                    const int  * min_index_candidates,
                                    float     * gpu_dist,
                                    int        query_index,
                                    int        query_nb,
                                    int         i,
                                    int         k,
                                    float      * knn_dist,
                                    int        * knn_index){

    // set up shared mem
    extern __shared__ float smem[];
    
    // copy distances and indexes to shared mem
    smem[threadIdx.x] = min_candidates[threadIdx.x];
    smem[threadIdx.x+blockDim.x] = min_index_candidates[threadIdx.x];

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

        knn_dist[(query_nb*i)+query_index] = smem[0];
        knn_index[(query_nb*i)+query_index] = min_index;

        // printf("placing %f at index: %d\n", smem[0], (k*query_index)+i);

        // set gpu_dist["min_index"] to 1
        gpu_dist[(min_index*query_nb)+query_index] = 1;

    }
}


// -- USED IN BASELINE APPROACH (it sucks) --
// count candidates in range [min, min+delta]
// every thread in the block adds 1 in smem then we sum all vals into global like in "histogram"
// do it using 1 block, each thread handles more queries in a for loop
__global__ void get_candidates(const float* gpu_dist,
                                const float* min_dist,
                                int          query_index,
                                int          ref_nb,
                                int          query_nb,
                                float        delta,
                                int        * candidates,
                                int        * count){

    // count candidates, every thread handles more queries and if condition is true flags corresponding cell and increase counter
    for(unsigned int i=0; i<ref_nb/blockDim.x; i++){
        if(gpu_dist[(query_nb*blockDim.x*i)+(threadIdx.x*query_nb)+query_index] < min_dist[query_index] + delta){
            candidates[(query_nb*blockDim.x*i)+(threadIdx.x*query_nb)+query_index] = 1;
            // printf("adding");
            atomicAdd(&count[query_index], 1);
        }
    }

}

// check if we have enough candidates in range [min, min+delta] for each query
__global__ void check_min_k(const int * count,
                            int       k,
                            int     * flag){
    
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(count[i] < k){
        *flag = 1;
    }


}

/**
 * Sort the distances stopping at K sorted elements
 * exploit mask: h_candidates
 * NOT WORKING
 */
void masked_insertion_sort(float *dist, int *index, int * mask, int length, int k, int query_index, int query_nb){
    // Initialize the first index
    index[0] = 0;

    int skip_counter = 0;

    // Go through all points
    for(unsigned int i=0; i<length; i++){

        float curr_dist = dist[i]; 
        int curr_index = index[i];

        // mask is a matrix of size query_nb * ref_nb
        if(mask[(i*query_nb)+query_index] == 0){
            skip_counter++;
            // printf("Skipping %d, ", i);
            continue;
        }

        // Skip the current value if its index is >= k and if it's higher the k-th already sorted smallest value
        if (i >= k && curr_dist >= dist[k-1]) {
            // printf("Skipping %d, ", i);
            continue;
        }

        

        // Shift values (and indexes) higher that the current distance to the right
        int j = min(i-skip_counter, k-1);
        while (j > 0 && dist[j-1] > curr_dist) {
            // printf("Shifting %d to %d\n", j-1, j);
            dist[j]  = dist[j-1];
            index[j] = index[j-1];
            --j;
        }

        // Write the current distance and index at their position
        dist[j]  = curr_dist;
        index[j] = curr_index; 

        // printf("passing %d\n", i);
    }
}

// -- USED FOR PICK_K ON GPU W INNER LOOP--
/*
    * Find min using reduction, then set corresponding value to 1 in gpu_dist and repeat k times
    * Handles distances and indexes
    * The idea behind this part is to exploit the fact that ref_nb is a perfect square, so we can match intra and inter part using same amount of threads.
    * Eg: ref_nb = 16384, we can have 128 threads per block and 128 blocks. And later block 0 will do final reduction on 128 values
    * 99% working, issue is that accessing global memory is slow, so we need to find a way to reduce the number of accesses
*/
__global__ void get_min4(float      * gpu_dist,
                         const int  * gpu_index,
                         int          query_index,
                         int          query_nb,
                         float      * min_candidates,
                         int        * min_index_candidates,
                         int          i,                        // #ith min             
                         float      * knn_dist,
                         int        * knn_index){

    // set up shared mem
    // 2 * blockDim * sizeof(float) for distances and indexes
    extern __shared__ float smem[];
    
    // Intrablock reduction phase

    // copy distances and indexes to shared mem
    smem[threadIdx.x] = gpu_dist[(query_nb*blockDim.x*blockIdx.x)+(threadIdx.x*query_nb)+query_index];
    smem[threadIdx.x+blockDim.x] = gpu_index[(query_nb*blockDim.x*blockIdx.x)+(threadIdx.x*query_nb)+query_index];

    __syncthreads();

    // if(smem[threadIdx.x] == 0){
    //     printf("smem[%d]: %f\n", threadIdx.x, smem[threadIdx.x]);
    //     printf("smem[%d]: %f\n", threadIdx.x+blockDim.x, smem[threadIdx.x+blockDim.x]);
    // }

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
        min_candidates[(blockIdx.x*query_nb)+query_index] = smem[0];
        min_index_candidates[(blockIdx.x*query_nb)+query_index] = smem[blockDim.x];
    }  

    __syncthreads();

    // Interblock reduction phase
    if (blockIdx.x == 0){

        // copy distances and indexes to shared mem
        smem[threadIdx.x] = min_candidates[(threadIdx.x*query_nb)+query_index];
        smem[threadIdx.x+blockDim.x] = min_index_candidates[(threadIdx.x*query_nb)+query_index];

        // if(smem[threadIdx.x] == 0){
        //     printf("smem[%d]: %f\n", threadIdx.x, smem[threadIdx.x]);
        //     printf("smem[%d]: %f\n", threadIdx.x+blockDim.x, smem[threadIdx.x+blockDim.x]);
        // }


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

            // printf("placing %f\n", smem[0]);
            knn_dist[(query_nb*i)+query_index] = smem[0];
            knn_index[(query_nb*i)+query_index] = min_index;

            // printf("placing %f at index: %d\n", smem[0], (k*query_index)+i);

            // set gpu_dist["min_index"] to 1
            gpu_dist[(min_index*query_nb)+query_index] = 1;

        }

        __syncthreads();

    }

}





/**
 * Selection sort the distances stopping at K sorted elements
 * exploit mask: h_candidates
 */
void selection_sort(float *dist, int *index, int *mask, int length, int k, int query_index, int query_nb, float *knn_dist, int *knn_index){

    for(int i=0; i<k; i++){
        float min_value = 1;
        int min_index = 0;

        for(int j=0; j<length; j++){
            if(mask[(j*query_nb)+query_index] == 0){
                continue;
            }
            if(dist[(j*query_nb)+query_index] < min_value){
                min_value = dist[(j*query_nb)+query_index];
                min_index = index[(j*query_nb)+query_index];
            }
        }

        // place values in knn_dist and knn_index
        knn_dist[(query_nb*i)+query_index] = min_value;
        knn_index[(query_nb*i)+query_index] = min_index;

        dist[(min_index*query_nb)+query_index] = 1;

    }
}


/**
 * Sort the distances stopping at K sorted elements
 * CPU version
 */
void  insertion_sort_on_matrix(float *dist, int *index, int length, int k, int query_index, int query_nb){

    // Initialise the first index
    index[0] = 0;

    // Go through all points
    for (int i=1; i<length; ++i) {

        // Store current distance and associated index
        float curr_dist  = dist[(i*query_nb)+query_index];
        int   curr_index = i;

        // Skip the current value if its index is >= k and if it's higher the k-th slready sorted mallest value
        if (i >= k && curr_dist >= dist[(k-1)*query_nb+query_index]) {
            continue;
        }

        // Shift values (and indexes) higher that the current distance to the right
        int j = min(i, k-1);
        while (j > 0 && dist[(j-1)*query_nb+query_index] > curr_dist) {
            dist[(j*query_nb)+query_index]  = dist[((j-1)*query_nb)+query_index];
            index[(j*query_nb)+query_index] = index[((j-1)*query_nb)+query_index];
            --j;
        }

        // Write the current distance and index at their position
        dist[(j*query_nb)+query_index]  = curr_dist;
        index[(j*query_nb)+query_index] = curr_index; 
    }
}

// The idea is to use a custom insertion sort that will only sort the k first elements on gpu where each thread will sort a query
// Since we have more than 1024 queries, we need to use several blocks to handle all queries
