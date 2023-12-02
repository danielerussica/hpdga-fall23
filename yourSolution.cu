#include <stdio.h>
#include <cuda.h>


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


bool your_solution_baseline(const float * ref,
                     int           ref_nb,
                     const float * query,
                     int           query_nb,
                     int           dim,
                     int           k,
                     float *       knn_dist,    // output fields
                     int *         knn_index) {

    uint64_t o_matrix_size = 1LL * ref_nb * query_nb * sizeof(float);

    float * cpu_dist   = (float*) malloc(o_matrix_size);
    float * h_gpu_dist = (float*) malloc(o_matrix_size);
    int   * h_gpu_index    = (int*)   malloc(o_matrix_size);

    int blockSize = dim;        // Number of threads per block
    int gridSize = ref_nb;      // Number of blocks

    // printf("blockSize: %d\n", blockSize);
    // printf("gridSize: %d\n", gridSize);

    // copy ref and query into cuda mem
    float   *d_ref, *d_query;
    float   *d_gpu_dist;
    int     *d_index;

    cudaMalloc(&d_ref, ref_nb * dim * sizeof(float));
    cudaMalloc(&d_query, ref_nb * dim * sizeof(float));

    cudaMalloc(&d_gpu_dist, o_matrix_size);
    cudaMalloc(&d_index, o_matrix_size);

    cudaMemcpy(  d_ref,   ref, ref_nb * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, ref_nb * dim * sizeof(float), cudaMemcpyHostToDevice);


    // Calculate the next power of 2 for dim
    int nextPow2 = 1;
    while (nextPow2 < dim) {
        nextPow2 <<= 1;
    }

    // Calculate the number of elements required in smem with padding
    int paddedDim = nextPow2/2;
    int smemSize = 3 * paddedDim * sizeof(float);
    // printf("paddedDim: %d\n", paddedDim);

    for(unsigned int query_index=0; query_index<query_nb; query_index++){
        // printf("Query %d\n", query_index);
        padded_cdist<<< gridSize, paddedDim, smemSize >>>(d_ref, ref_nb, d_query, query_nb, dim, paddedDim, query_index, d_gpu_dist, d_index);
    }

    // mem copy back to cpu
    // very expensive and can be done in parallel with k selection, as long as memcopy is done before actual selection
    cudaMemcpy(h_gpu_dist, d_gpu_dist, o_matrix_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gpu_index, d_index, o_matrix_size, cudaMemcpyDeviceToHost);

    cudaFree(d_ref);
    cudaFree(d_query);


    // K SELECTION --------------------------------------------------------------------------------------------------------------------------------

    int blockSize2 = 64;        // Test to find the best value theoretically we want second level to use at least 32 blocks to avoid inactive threads in block
    int gridSize2 = ref_nb/blockSize2;

    // allocate cuda mem
    float *d_min_distances;
    float *d_min_dist;

    cudaMalloc(&d_min_distances, query_nb * gridSize2 * sizeof(float));
    cudaMalloc(&d_min_dist, query_nb * sizeof(float));
    

    // printf("\n\nSearching for min\n");
    // printf("blockSize: %d\n", blockSize2);
    // printf("gridSize: %d\n", gridSize2);
    // get min for each query
    for(unsigned int query_index=0; query_index<query_nb; query_index++){
        get_min_intrablock<<< gridSize2, blockSize2, blockSize2 * sizeof(float) >>>(d_gpu_dist, query_index, query_nb, d_min_distances);
        get_min_interblock<<< 1, gridSize2, gridSize2 * sizeof(float) >>>(d_min_distances, query_index, d_min_dist);
    }

    // -------------- PART 2.5: add delta and count candidates -------------------------------------------------------------------------------------

    // add delta
    float delta = 0.1;

    int *d_candidates;
    int *d_count;
    int *d_flag;

    cudaMalloc(&d_candidates, query_nb * ref_nb * sizeof(float));
    cudaMalloc(&d_count, query_nb * sizeof(int));

    cudaMemset(d_candidates, 0, query_nb * ref_nb * sizeof(float));
    cudaMemset(d_count, 0, query_nb * sizeof(int));

    int *h_count = (int*) malloc(query_nb * sizeof(int));
    int *h_candidates = (int*) calloc(query_nb * ref_nb, sizeof(int));
    int *h_flag = (int*) malloc(sizeof(int));

    int blockSize3 = 1024;
    int gridSize3 = query_nb/blockSize3;       

    int collected_candidates = 0;
    while(!collected_candidates){
        // printf("\n\nSearching for candidates\n");
        for(unsigned int query_index=0; query_index<query_nb; query_index++){
            get_candidates<<< 1, blockSize2 >>> (d_gpu_dist, d_min_dist, query_index, ref_nb, query_nb, delta, d_candidates, d_count);
        }

        cudaDeviceSynchronize();

        check_min_k<<<gridSize3, blockSize3>>>(d_count, k, d_flag);


        *h_flag = 0;
        cudaMemcpy(h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);

        if(*h_flag == 0){
            collected_candidates = 1;
        }
        else{
            delta += delta/2;
            cudaMemset(d_count, 0, query_nb * sizeof(int));
        }

    }

    cudaMemcpy(h_count, d_count, query_nb * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_candidates, d_candidates, query_nb * ref_nb * sizeof(int), cudaMemcpyDeviceToHost);


    // printf("number of candidates:\n");
    // for(unsigned int i=0; i<query_nb; i++){
    //     printf("%d, ", h_count[i]);
    //     // assert(h_count[i] >= k);
    // }
    // printf("\n");

    // do insertion sort on cpu exploiting h_candidates
    
    for(unsigned int query_index=0; query_index<query_nb; query_index++){

        // do insertion sort
        // masked_insertion_sort(dist, index, h_candidates, ref_nb, k, query_index, query_nb);
        // insertion_sort(dist, index, ref_nb, k);
        selection_sort(h_gpu_dist, h_gpu_index, h_candidates, ref_nb, k, query_index, query_nb, knn_dist, knn_index);
        
        // print first k elements
        // printf("Query %d: ", query_index);
        // for(unsigned int i=0; i<k; i++){
        //     printf("%f || ", dist[i]);
        // }
        // printf("\n");

        // copy k smallest distances and their associated index (done in selection_sort)
        // for (int j=0; j<k; ++j) {
        //     knn_dist[j * query_nb + query_index]  = dist[j];
        //     knn_index[j * query_nb + query_index] = index[j];
        // }
    }

   return true;
}






// solution to test if computing min and counting candidates to help cpu actually helps the cpu
bool your_solution_only_dist(const float * ref,
                     int           ref_nb,
                     const float * query,
                     int           query_nb,
                     int           dim,
                     int           k,
                     float *       knn_dist,    // output fields
                     int *         knn_index) {

    uint64_t o_matrix_size = 1LL * ref_nb * query_nb * sizeof(float);

    float * cpu_dist   = (float*) malloc(o_matrix_size);
    float * h_gpu_dist = (float*) malloc(o_matrix_size);
    int   * h_gpu_index    = (int*)   malloc(o_matrix_size);

    int blockSize = dim;        // Number of threads per block
    int gridSize = ref_nb;      // Number of blocks

    // printf("blockSize: %d\n", blockSize);
    // printf("gridSize: %d\n", gridSize);

    // copy ref and query into cuda mem
    float   *d_ref, *d_query;
    float   *d_gpu_dist;
    int     *d_index;

    cudaMalloc(&d_ref, ref_nb * dim * sizeof(float));
    cudaMalloc(&d_query, ref_nb * dim * sizeof(float));

    cudaMalloc(&d_gpu_dist, o_matrix_size);
    cudaMalloc(&d_index, o_matrix_size);

    cudaMemcpy(  d_ref,   ref, ref_nb * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, ref_nb * dim * sizeof(float), cudaMemcpyHostToDevice);


    // Calculate the next power of 2 for dim
    int nextPow2 = 1;
    while (nextPow2 < dim) {
        nextPow2 <<= 1;
    }

    // Calculate the number of elements required in smem with padding
    int paddedDim = nextPow2/2;
    int smemSize = 3 * paddedDim * sizeof(float);
    // printf("paddedDim: %d\n", paddedDim);

    for(unsigned int query_index=0; query_index<query_nb; query_index++){
        // printf("Query %d\n", query_index);
        padded_cdist<<< gridSize, paddedDim, smemSize >>>(d_ref, ref_nb, d_query, query_nb, dim, paddedDim, query_index, d_gpu_dist, d_index);
    }

    // mem copy back to cpu
    // very expensive and can be done in parallel with k selection, as long as memcopy is done before actual selection
    cudaMemcpy(h_gpu_dist, d_gpu_dist, o_matrix_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gpu_index, d_index, o_matrix_size, cudaMemcpyDeviceToHost);

    cudaFree(d_ref);
    cudaFree(d_query);


    // K SELECTION ON CPU --------------------------------------------------------------------------------------------------------------------------------

    for(int i=0; i<query_nb; i++){
        insertion_sort_on_matrix(h_gpu_dist, h_gpu_index, ref_nb, k, i, query_nb);

        // Copy k smallest distances and their associated index
        for (int j=0; j<k; ++j) {
        knn_dist[j * query_nb + i]  = h_gpu_dist[(j*query_nb)+i];
        knn_index[j * query_nb + i] = h_gpu_index[(j*query_nb)+i];
        }
    }

   return true;
}