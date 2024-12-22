import copy
import math

from lab_2.src.tools.clusterers.clusterer import Clusterer
from lab_2.src.tools import ploter


class KNearestNeighbors(Clusterer):

    def __init__(self, sample, clusters):
        super().__init__(sample)
        self._clusters = clusters
        self._new_clusters = copy.deepcopy(clusters)
        self._k = None

    def clusterize(self):
        for i in self._sample:
            self._new_clusters[self._calculate_cluster_id(i)].append(i)
        return self._new_clusters

    def _calculate_cluster_id(self, point):
        point_neighbors = self._find_k_nearest_neighbors(self._k, point)
        cluster_counts = {}
        for neighbor in point_neighbors:
            c_id = neighbor["c_id"]
            cluster_counts[c_id] = cluster_counts.get(c_id, 0) + 1
        return max(cluster_counts, key=cluster_counts.get)

    def _find_k_nearest_neighbors(self, k, point):
        neighbors = []
        for c_id in range(len(self._new_clusters)):
            for p_id in range(len(self._new_clusters[c_id])):
                distance = self._euclid_distance(point, self._new_clusters[c_id][p_id])
                neighbors.append({"c_id": c_id, "p_id": p_id, "distance": distance})
        return sorted(neighbors, key=lambda x: x["distance"])[:k]

    def _euclid_distance(self, point_1, point_2):
        d_part = []
        for c in range(len(point_1)):
            d_part.append((point_1[c] - point_2[c])**2)
        return math.sqrt(sum(d_part))

    def _ready(self):
        return self._sample is not None and self._new_clusters is not None and self._k >= 1

    def set_k(self, k):
        self._k = k

    def info(self):
        ploter.dots(self._new_clusters)
