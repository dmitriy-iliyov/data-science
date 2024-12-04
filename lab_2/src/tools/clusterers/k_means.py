import math
import random

from src.tools.clusterers.clusterer import Clusterer
from src.tools import ploter


class KMeans(Clusterer):

    def __init__(self, sample):
        super().__init__(sample)
        self._cluster_count = None
        self._centroids = []

    def clusterize(self):
        if self._ready():
            self._update_clusters()
            previous_centroids = [centroid[:] for centroid in self._centroids]
            while True:
                self._update_clusters()
                self._update_centroids()
                if self._changed(previous_centroids):
                    return self._clusters
                previous_centroids = [centroid[:] for centroid in self._centroids]

    def _update_clusters(self):
        self._clusters = [[] for _ in range(self._cluster_count)]
        for i in self._sample:
            cluster_id = self._calculate_cluster_id(i)
            self._clusters[cluster_id].append(i)

    def _update_centroids(self):
        for i in range(self._cluster_count):
            self._calculate_cluster_centroids(i)

    def _calculate_cluster_centroids(self, i):
        cluster_length = len(self._clusters[i])
        if cluster_length == 0:
            return
        cluster_sum = [0 for _ in range(len(self._clusters[0][0]))]
        for point in self._clusters[i]:
            for j in range(len(point)):
                cluster_sum[j] += point[j]
        self._centroids[i] = [x / cluster_length for x in cluster_sum]

    def _calculate_cluster_id(self, coordinates):
        cluster_id = None
        min_distance = float('inf')
        for centroid in self._centroids:
            d_part = []
            for i in range(len(coordinates)):
                d_part.append((coordinates[i] - centroid[i])**2)
            distance = math.sqrt(sum(d_part))
            if distance < min_distance:
                min_distance = distance
                cluster_id = self._centroids.index(centroid)
        return cluster_id

    def _changed(self, previous_centroids):
        for i in range(self._cluster_count):
            if previous_centroids[i] != self._centroids[i]:
                return False
        return True

    def set_cluster_count(self, cluster_count):
        self._cluster_count = cluster_count
        self._clusters = [[] for _ in range(cluster_count)]

    def set_cluster_centroids(self, centroids=None):
        if centroids:
            self._centroids = centroids
        elif self._cluster_count is not None:
            self._centroids = random.sample(self._sample, self._cluster_count)

    def _ready(self):
        return self._cluster_count is not None and self._centroids is not None

    def info(self):
        ploter.dots(self._clusters)
