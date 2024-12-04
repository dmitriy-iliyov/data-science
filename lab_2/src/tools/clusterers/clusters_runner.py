import random
from src.tools.clusterers.k_means import KMeans
from src.tools.clusterers.k_nearest_neighbors import KNearestNeighbors


def rand_coordinates(n, _min=-10, _max=10):
    return [[random.uniform(_min, _max), random.uniform(_min, _max)] for i in range(n)]


coordinates = rand_coordinates(20)
kM = KMeans(coordinates)
kM.set_cluster_count(3)
kM.set_cluster_centroids()
cl = kM.clusterize()
kM.info()

coordinates_2 = rand_coordinates(100)
kNN = KNearestNeighbors(coordinates_2, cl)
kNN.set_k(3)
cl2 = kNN.clusterize()
kNN.info()

sample = []
for i in cl2:
    for j in i:
        sample.append(j)
kM2 = KMeans(sample)
kM2.set_cluster_count(3)
kM2.set_cluster_centroids()
cl_3 = kM2.clusterize()
kM2.info()
