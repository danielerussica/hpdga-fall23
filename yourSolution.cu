#include <stdio.h>
#include <cuda.h>




bool your_solution(const float * ref,
                     int           ref_nb,
                     const float * query,
                     int           query_nb,
                     int           dim,
                     int           k,
                     float *       knn_dist,    // output fields
                     int *         knn_index) {

   // CLONA TABELLA DEI VETTORI IN CUDAMEM, ALLOCA SPAZIO PER OUTPUT

   // copy ref and query into cuda mem
   float *d_ref, *d_query;
   
   cudaMalloc(&d_ref, ref_nb * dim * sizeof(float));
   cudaMalloc(&d_query, ref_nb * dim * sizeof(float));

   cudaMemcpy(d_ref, ref, ref_nb * dim * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_query, query, ref_nb * dim * sizeof(float), cudaMemcpyHostToDevice);
   

   // Allocate local array to store ALL the distances / indexes for a given query point 
   float *dist  = (float *) malloc(ref_nb * sizeof(float));
   int   *index = (int *)   malloc(ref_nb * sizeof(int));

    // Allocation checks
   if (!dist || !index) {
      printf("Memory allocation error\n");
      free(dist);
      free(index);
      return false;
   }

   // SOLUTION STARTS HERE





   return true;
}