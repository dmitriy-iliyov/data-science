import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


from src.tools import parser


class CryptoClusterizator:

    def __init__(self, n_cluster, coin_names=None):
        self._coin_names = coin_names
        self._coin_data = self._from_csv()
        print(self._coin_data[self._coin_names[0]].keys())
        self._n_cluster = n_cluster
        self._kmeans = KMeans(n_clusters=n_cluster)

    def clusterize(self, key_1, key_2):
        sample = [
            [self._coin_data[key][key_1].iloc[0], self._coin_data[key][key_2].iloc[0]]
            for key in self._coin_data.keys()
        ]
        self._kmeans.fit(sample)
        self.info(key_1, key_2)

    def _from_csv(self):
        self._coin_names = [file.split('_')[0] for file in os.listdir('files') if file.endswith('_market_data.csv')]
        coin_data = {}
        for name in self._coin_names:
            try:
                filename = name.lower() + '_market_data.csv'
                file_path = os.path.join('files', filename)
                if os.path.exists(file_path):
                    coin_data[name] = pd.read_csv(file_path)
                else:
                    print(f"file for {name} not found")
            except Exception as e:
                print(f"Error reading file for {name}: {e}")
                continue
        return coin_data

    def info(self, key_1, key_2):
        labels = self._kmeans.labels_
        cluster_info = {}
        for i, label in enumerate(labels):
            coin_name = self._coin_names[i]
            if label not in cluster_info:
                cluster_info[label] = []
            cluster_info[label].append(coin_name)
        self.dots(cluster_info, key_1, key_2)

    def dots(self, cluster_info, key_1, key_2):
        color_names = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        for cluster_id, coins in cluster_info.items():
            data_points = []
            for coin_name in coins:
                data_points.append([self._coin_data[coin_name][key_1].iloc[0], self._coin_data[coin_name][key_2].iloc[0]])
            color = color_names[cluster_id % len(color_names)]
            for point in data_points:
                plt.scatter(point[0], point[1], color=color, marker='o', s=10)
        plt.xlabel(key_1)
        plt.ylabel(key_2)
        plt.title(f"Clusters based on {key_1} and {key_2}")
        plt.show()
