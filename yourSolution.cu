#include "kernels.cu"
#include "old_kernels.cu"
#include <iostream>

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
    cudaMalloc(&d_query, query_nb * dim * sizeof(float));

    cudaMalloc(&d_gpu_dist, o_matrix_size);
    cudaMalloc(&d_index, o_matrix_size);

    cudaMemcpy(  d_ref,   ref, ref_nb * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, query_nb * dim * sizeof(float), cudaMemcpyHostToDevice);


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
    cudaMalloc(&d_query, query_nb * dim * sizeof(float));

    cudaMalloc(&d_gpu_dist, o_matrix_size);
    cudaMalloc(&d_index, o_matrix_size);

    cudaMemcpy(  d_ref,   ref, ref_nb * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, query_nb * dim * sizeof(float), cudaMemcpyHostToDevice);


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


// NOT WORKING YET
// solution that compute dists on gpu and compact "sparse array" on gpu using stream compaction
bool your_solution_stream_compaction(const float * ref,
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
    int  * h_gpu_prefix_sum    = (int*)   malloc(o_matrix_size);

    int blockSize = dim;        // Number of threads per block
    int gridSize = ref_nb;      // Number of blocks

    // printf("blockSize: %d\n", blockSize);
    // printf("gridSize: %d\n", gridSize);

    // copy ref and query into cuda mem
    float   *d_ref, *d_query;
    float   *d_gpu_dist;
    int     *d_index;
    int     *d_valid;
    int     *d_prefix_sum;

    cudaMalloc(&d_ref, ref_nb * dim * sizeof(float));
    cudaMalloc(&d_query, query_nb * dim * sizeof(float));

    cudaMalloc(&d_gpu_dist, o_matrix_size);
    cudaMalloc(&d_index, o_matrix_size);
    cudaMalloc(&d_valid, o_matrix_size);
    cudaMalloc(&d_prefix_sum, query_nb * sizeof(int));

    cudaMemcpy(  d_ref,   ref, ref_nb * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, query_nb * dim * sizeof(float), cudaMemcpyHostToDevice);


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
        padded_cdist_with_valid<<< gridSize, paddedDim, smemSize >>>(d_ref, ref_nb, d_query, query_nb, dim, paddedDim, query_index, d_gpu_dist, d_index, d_valid);
    }

    cudaFree(d_ref);
    cudaFree(d_query);

    int blockSize_prefix_sum = 1024;

    // stream compaction
    for(unsigned int query_index=0; query_index<query_nb; query_index++){
        prefix_sum<<< ref_nb/blockSize_prefix_sum, blockSize_prefix_sum, blockSize_prefix_sum * sizeof(int) >>>(d_valid, query_nb, query_index, d_prefix_sum);
    }

    // print out prefix sum in a file
    // cudaMemcpy(h_gpu_prefix_sum, d_valid, o_matrix_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gpu_prefix_sum, d_prefix_sum, o_matrix_size, cudaMemcpyDeviceToHost);

    FILE *f = fopen("prefix_sum.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    for(unsigned int i=0; i<10; i++){
        for(unsigned int j=0; j<ref_nb; j++){
            fprintf(f, "%d ", h_gpu_prefix_sum[(query_nb*j)+i]);
        }
        fprintf(f, "\n");
    }

    fclose(f);

    

   return true;
}


bool your_solution_pick_k_on_gpu(const float * ref,
                     int           ref_nb,
                     const float * query,
                     int           query_nb,
                     int           dim,
                     int           k,
                     float *       knn_dist,    // output fields
                     int *         knn_index) {



    // Solution that naively pick k-smallest on gpu using reduction
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
    cudaMalloc(&d_query, query_nb * dim * sizeof(float));

    cudaMalloc(&d_gpu_dist, o_matrix_size);
    cudaMalloc(&d_index, o_matrix_size);

    cudaMemcpy(  d_ref,   ref, ref_nb * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, query_nb * dim * sizeof(float), cudaMemcpyHostToDevice);


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

    cudaFree(d_ref);
    cudaFree(d_query);


    // K SELECTION --------------------------------------------------------------------------------------------------------------------------------

    int blockSize2 = 64;        // Test to find the best value theoretically we want second level to use at least 32 blocks to avoid inactive threads in block
    int gridSize2 = ref_nb/blockSize2;

    // allocate cuda mem
    float *d_min_distances;
    int *d_min_indexes;

    float *d_knn_dist;
    int *d_knn_index;
    

    cudaMalloc(&d_min_distances, query_nb * gridSize2 * sizeof(float));
    cudaMalloc(&d_min_indexes, query_nb * gridSize2 * sizeof(int));

    cudaMalloc(&d_knn_dist, query_nb * k * sizeof(float));
    cudaMalloc(&d_knn_index, query_nb * k * sizeof(int));
    
    // to do: try to implement stream version of this
    for(unsigned int i=0; i<k; i++){
        for(unsigned int query_index=0; query_index<query_nb; query_index++){
            get_min_intrablock2<<< gridSize2, blockSize2, 2 * blockSize2 * sizeof(float) >>>(d_gpu_dist, d_index, query_index, query_nb, d_min_distances, d_min_indexes);
            get_min_interblock2<<< 1, gridSize2, 2 * gridSize2 * sizeof(float) >>>(d_min_distances, d_min_indexes, d_gpu_dist, query_index, query_nb, i, k, d_knn_dist, d_knn_index);
        }
    }

    // mem copy back to cpu
    cudaMemcpy(knn_dist, d_knn_dist, query_nb * k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(knn_index, d_knn_index, query_nb * k * sizeof(int), cudaMemcpyDeviceToHost);

    return true;
}



bool your_solution_pick_k_on_gpu_w_stream(const float * ref,
                     int           ref_nb,
                     const float * query,
                     int           query_nb,
                     int           dim,
                     int           k,
                     float *       knn_dist,    // output fields
                     int *         knn_index) {



    // Solution that naively pick k-smallest on gpu using reduction
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
    cudaMalloc(&d_query, query_nb * dim * sizeof(float));

    cudaMalloc(&d_gpu_dist, o_matrix_size);
    cudaMalloc(&d_index, o_matrix_size);

    cudaMemcpy(  d_ref,   ref, ref_nb * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, query_nb * dim * sizeof(float), cudaMemcpyHostToDevice);


    // COSINE DISTANCE --------------------------------------------------------------------------------------------------------------------------------

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

    cudaDeviceSynchronize();


    // K SELECTION --------------------------------------------------------------------------------------------------------------------------------

    int blockSize2 = 64;        // (tested: 64>512 with param1) Test to find the best value theoretically we want second level to use at least 32 blocks to avoid inactive threads in block
    int gridSize2 = ref_nb/blockSize2;

    // allocate cuda mem
    float *d_min_distances;
    int *d_min_indexes;

    float *d_knn_dist;
    int *d_knn_index;
    

    cudaMalloc(&d_min_distances, query_nb * gridSize2 * sizeof(float));
    cudaMalloc(&d_min_indexes, query_nb * gridSize2 * sizeof(int));

    cudaMalloc(&d_knn_dist, query_nb * k * sizeof(float));
    cudaMalloc(&d_knn_index, query_nb * k * sizeof(int));
    
    // create n streams, divide gpu_dist and gpu_index into n parts and do k selection on each stream
    int n_streams = 128;
    cudaStream_t stream[n_streams];

    for (int i = 0; i < n_streams; i++) {
        cudaStreamCreate(&stream[i]);
    }

    // array of streams

    int offset = query_nb/n_streams;

    for(unsigned int i=0; i<k; i++){
        for(unsigned int query_index=0; query_index<query_nb/n_streams; query_index++){
            for(int j=0; j<n_streams; j++){
                
                get_min_intrablock3<<< gridSize2, blockSize2, 2 * blockSize2 * sizeof(float), stream[j] >>>(d_gpu_dist, d_index, query_index, offset*j, query_nb, d_min_distances, d_min_indexes);
                get_min_interblock3<<< 1, gridSize2, 2 * gridSize2 * sizeof(float), stream[j] >>>(d_min_distances, d_min_indexes, d_gpu_dist, query_index, offset*j, query_nb, i, k, d_knn_dist, d_knn_index);
            }
        }
    }

    // mem copy back to cpu
    cudaMemcpy(knn_dist, d_knn_dist, query_nb * k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(knn_index, d_knn_index, query_nb * k * sizeof(int), cudaMemcpyDeviceToHost);


    // //fprintf knn_dist and knn_index to file
    // FILE *f = fopen("knn_dist.txt", "w");
    // if (f == NULL)
    // {
    //     printf("Error opening file!\n");
    //     exit(1);
    // }

    // for(unsigned int i=0; i<query_nb; i++){
    //     for(unsigned int j=0; j<k; j++){
    //         fprintf(f, "%f ", knn_dist[(query_nb*j)+i]);
    //     }
    //     fprintf(f, "\n");
    // }

    // fclose(f);

    // f = fopen("knn_index.txt", "w");
    // if (f == NULL)
    // {
    //     printf("Error opening file!\n");
    //     exit(1);
    // }

    // for(unsigned int i=0; i<query_nb; i++){
    //     for(unsigned int j=0; j<k; j++){
    //         fprintf(f, "%d ", knn_index[(query_nb*j)+i]);
    //     }
    //     fprintf(f, "\n");
    // }

    // fclose(f);

    // free cuda mem
    cudaFree(d_ref);
    cudaFree(d_query);
    cudaFree(d_min_distances);
    cudaFree(d_min_indexes);
    cudaFree(d_knn_dist);
    cudaFree(d_knn_index);
    cudaFree(d_gpu_dist);
    cudaFree(d_index);



    return true;
}

// get_min4 is broken...
bool ys_pick_kgpu_innerfor(const float * ref,
                     int           ref_nb,
                     const float * query,
                     int           query_nb,
                     int           dim,
                     int           k,
                     float *       knn_dist,    // output fields
                     int *         knn_index) {



    // Solution that naively pick k-smallest on gpu using reduction
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
    cudaMalloc(&d_query, query_nb * dim * sizeof(float));

    cudaMalloc(&d_gpu_dist, o_matrix_size);
    cudaMalloc(&d_index, o_matrix_size);

    cudaMemcpy(  d_ref,   ref, ref_nb * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, query_nb * dim * sizeof(float), cudaMemcpyHostToDevice);


    // Calculate the next power of 2 for dim
    int nextPow2 = 1;
    while (nextPow2 < dim) {
        nextPow2 <<= 1;
    }

    // Calculate the number of elements required in smem with padding
    int paddedDim = nextPow2/2;
    int smemSize = 3 * paddedDim * sizeof(float);
    // printf("paddedDim: %d\n", paddedDim);

    padded_cdist2_innerfor<<< gridSize, paddedDim, smemSize >>>(d_ref, ref_nb, d_query, query_nb, dim, paddedDim, d_gpu_dist, d_index);

    cudaFree(d_ref);
    cudaFree(d_query);


    // K SELECTION --------------------------------------------------------------------------------------------------------------------------------

    int blockSize2 = int(sqrt(double(ref_nb)));
    int gridSize2 = blockSize2;

    // printf("blockSize: %d\n", blockSize2);

    // allocate cuda mem
    float *d_min_distances;
    int *d_min_indexes;

    float *d_knn_dist;
    int *d_knn_index;
    

    cudaMalloc(&d_min_distances, query_nb * gridSize2 * sizeof(float));
    cudaMalloc(&d_min_indexes, query_nb * gridSize2 * sizeof(int));

    cudaMalloc(&d_knn_dist, query_nb * k * sizeof(float));
    cudaMalloc(&d_knn_index, query_nb * k * sizeof(int));
    
    for(unsigned int i=0; i<k; i++){
        for(unsigned int query_index=0; query_index<query_nb; query_index++){
            get_min4<<< blockSize2, gridSize2, 2 * gridSize2 * sizeof(float) >>>(d_gpu_dist, d_index, query_index, query_nb, d_min_distances, d_min_indexes, i, d_knn_dist, d_knn_index);
        }
    }


    


    // mem copy back to cpu
    cudaMemcpy(knn_dist, d_knn_dist, query_nb * k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(knn_index, d_knn_index, query_nb * k * sizeof(int), cudaMemcpyDeviceToHost);



    // //fprintf knn_dist and knn_index to file
    FILE *f = fopen("knn_dist.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    for(unsigned int i=0; i<query_nb; i++){
        for(unsigned int j=0; j<k; j++){
            fprintf(f, "%f ", knn_dist[(query_nb*j)+i]);
        }
        fprintf(f, "\n");
    }

    fclose(f);

    f = fopen("knn_index.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    for(unsigned int i=0; i<query_nb; i++){
        for(unsigned int j=0; j<k; j++){
            fprintf(f, "%d ", knn_index[(query_nb*j)+i]);
        }
        fprintf(f, "\n");
    }

    fclose(f);

    return true;
}


bool ys_for_param2(const float * ref,
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
    std::cout << "Total GPU Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB\n";

    uint64_t o_matrix_size = 1L * ref_nb * query_nb * sizeof(float);

    uint64_t batches = ceil((double)2*o_matrix_size/(deviceProp.totalGlobalMem));
    std::cout << "batches = " << o_matrix_size << "/" << deviceProp.totalGlobalMem << " = " << batches << "\n";  

    int gridSize = ref_nb;      // Number of blocks for cdist

    for(unsigned int batch = 0; batch<batches; batch++){
        
        printf("batch %d\n", batch);

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

        cdist3<<< gridSize, paddedDim, smemSize >>>(d_ref, ref_nb, d_query, query_nb/batches, dim, paddedDim, batch*(query_nb/batches),d_gpu_dist, d_index);

        printf("end cdist\n");

        // batch k selection ----------------------------------------------------------------------------------------------------------------------
        int blockSize2 = 1024;
        int gridSize2 = ref_nb/blockSize2;

        nextPow2 = 1;
        while (nextPow2 < gridSize2) {
            nextPow2 <<= 1;
        }

        int interblockGridSize = nextPow2;
        printf("paddedGridSize2: %d\n", interblockGridSize);

        // allocate cuda mem
        float *d_min_distances;
        int *d_min_indexes;

        cudaMalloc(&d_min_distances, query_nb/batches * gridSize2 * sizeof(float));
        cudaMalloc(&d_min_indexes,   query_nb/batches * gridSize2 * sizeof(int));

        // to do: try to implement stream version of this
        for(unsigned int i=0; i<k; i++){
            for(unsigned int query_index=0; query_index<query_nb/batches; query_index++){
                get_min_intrablock3<<< gridSize2, blockSize2, 2 * blockSize2 * sizeof(float) >>>(d_gpu_dist, d_index, query_index, 0, query_nb/batches, d_min_distances, d_min_indexes);
                // temp_print_candidates<<< 1, 1>>>(d_min_distances, gridSize2, query_nb/batches); 
                get_min_interblock5<<< 1, interblockGridSize, 2 * interblockGridSize * sizeof(float) >>>(d_min_distances, d_min_indexes, gridSize2, d_gpu_dist, query_index,  query_nb/batches, batch*(query_nb/batches),batches, i, k, d_knn_dist, d_knn_index);
            }
        }

        printf("end k selection\n");

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
