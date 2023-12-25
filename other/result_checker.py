# create matrix of ground truth dist and index

import os
import sys
import matplotlib.pyplot as plt

# gt results
with open('gt_dist.txt') as f:
    gt_dist = [[float(x) for x in line.split()] for line in f]

with open('gt_index.txt') as f:
    gt_index = [[int(x) for x in line.split()] for line in f]

with open('knn_dist.txt') as f:
    gpu_dist = [[float(x) for x in line.split()] for line in f]

with open('knn_index.txt') as f:
    gpu_index = [[int(x) for x in line.split()] for line in f]


# check element by element of dist and print out the index if there is a mismatch
dist_mismatch = 0
precision = 0.001
for i in range(len(gt_dist)):
    for j in range(len(gt_dist[i])):
        if abs(gt_dist[i][j] - gpu_dist[i][j]) > precision:
            print("dist mismatch at i = %d, j = %d" % (i, j))
            print("gt_dist = %f, gpu_dist = %f" % (gt_dist[i][j], gpu_dist[i][j]))
            print("gt_index = %d, gpu_index = %d" % (gt_index[i][j], gpu_index[i][j]))
            print("")
            dist_mismatch += 1


# same for indexes
index_mismatch = 0
for i in range(len(gt_index)):
    for j in range(len(gt_index[i])):
        if gt_index[i][j] != gpu_index[i][j]:
            print("index mismatch at i = %d, j = %d" % (i, j))
            print("gt_dist = %f, gpu_dist = %f" % (gt_dist[i][j], gpu_dist[i][j]))
            print("gt_index = %d, gpu_index = %d" % (gt_index[i][j], gpu_index[i][j]))
            print("")
            index_mismatch += 1

print("\n\ndist_mismatch = %d, index_mismatch = %d" % (dist_mismatch, index_mismatch))
print("percentage of dist mismatch = %f" % (float(dist_mismatch + index_mismatch) / (len(gt_dist) * len(gt_dist[0]))))
print("percentage of index mismatch = %f" % (float(index_mismatch) / (len(gt_dist) * len(gt_dist[0]))))