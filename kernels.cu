#include <stdio.h>
#include <cuda.h>





// -- CURRENT VERSION --
/*
    * Compute the cosine distance between two vectors
    * inspired from Cuda webinar on reduction kernel04
    * Half the number of threads per block compared to previous version
    * Use padding to handle non Po2 dimensions
    * This version can handle more than 1280 dimensions (max 2048)
*/
__global__ void padded_cdist(const float   * ref,
                        int           ref_nb,
                        const float * query,
                        int           query_nb,
                        int           dim,
                        int           paddedDim,
                        int           query_index,
                        float       * d_gpu_dist,
                        int         * d_index){

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
        d_gpu_dist[(query_nb*blockIdx.x)+query_index] = smem[0] / (sqrt(smem[blockDim.x]) * sqrt(smem[2*blockDim.x]));
        d_index[(query_nb*blockIdx.x)+query_index] = blockIdx.x;
    }
}

// -- CURRENT VERSION --
// -- FOR LOOP IN KERNEL VERSION --
__global__ void padded_cdist2_innerfor(const float   * ref,
                        int           ref_nb,
                        const float * query,
                        int           query_nb,
                        int           dim,
                        int           paddedDim,
                        float       * d_gpu_dist,
                        int         * d_index){

    // we need 3 * paddedDim * sizeof(float) shared memory
    extern __shared__ float smem[];

    // unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int tid = threadIdx.x;

    for(unsigned int query_index=0; query_index<query_nb; query_index++){
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
            d_gpu_dist[(query_nb*blockIdx.x)+query_index] = smem[0] / (sqrt(smem[blockDim.x]) * sqrt(smem[2*blockDim.x]));
            d_index[(query_nb*blockIdx.x)+query_index] = blockIdx.x;
        }
    }

}

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
                                    float      * min_candidates,
                                    int        * min_index_candidates){

    // set up shared mem
    // 2 * blockDim * sizeof(float) for distances and indexes
    extern __shared__ float smem[];
    
    for(unsigned int query_index=0; query_index<query_nb; query_index++){
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

__global__ void gpu_custom_insertion_sort(float *dist, int *index, int length, int k, int query_nb, int offset, float *knn_dist, int *knn_index){
    
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
        knn_dist[(query_nb * i) + tid + offset] = dist[(i * query_nb) + tid];
        knn_index[(query_nb * i) + tid + offset] = index[(i * query_nb) + tid];
    }
}