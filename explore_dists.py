from matplotlib import pyplot as plt


# calculate average over all lines in file h_dists.txt
def avg_dist():
    with open("h_gpu_dist.txt", "r") as f:
        dists = [float(line.strip()) for line in f.readlines()]
        return sum(dists) / len(dists)
    
# print("Average distance: ", avg_dist())
# Average distance:  0.7503219915161117

# plot histogram of average distances
# save it to a file
# with open("h_gpu_dist.txt", "r") as f:
#     dists = [float(line.strip()) for line in f.readlines()]
#     plt.hist(dists, bins=100)
#     plt.title("Histogram of average distances")
#     plt.xlabel("Distance")
#     plt.ylabel("Frequency")
    
#     plt.savefig("h_gpu_dist_hist.png")


# plot histogram of distances query 0 vs all reference points
# save it to a file
# h_gpu_dist.txt is a csv
# 0.4, 0.6, 0.2, 0.3, 0.1, 0.5, 0.7, 0.8, 0.9, 0.0 ecc...
# with open("h_gpu_dist.txt", "r") as f:
#     # get first line
#     line = f.readline()
#     # split it by commas
#     dists = line.split(",")
#     # convert to float
#     dists = [float(dist) for dist in dists]

#     plt.hist(dists, bins=100)
#     plt.title("Histogram of distances query 0 vs all reference points")
#     plt.xlabel("Distance")
#     plt.ylabel("Frequency")

#     # zoom in around 0.75
#     plt.xlim(0.50, 0.9)
    
#     plt.savefig("h_gpu_dist_hist.png")


# parse h_gpu_dist.txt a csv
# plot histogram
with open("h_gpu_dist.txt", "r") as f:
    # get first line
    line = f.readline()
    # split it by commas
    dists = line.split(",")
    # convert to float
    dists = [int(dist) for dist in dists]

    # print(dists)
    
    plt.hist(dists)
    plt.title("n of candidates selected")
    plt.xlabel("n")
    plt.ylabel("elements")

    plt.xlim(90, 110)
    plt.savefig("delta_h_gpu_dist_hist.png")

# print min and max
print(min(dists))
print(max(dists))

# print avg
print(sum(dists) / len(dists))