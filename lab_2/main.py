import os

from src.crypto_clusterizator import CryptoClusterizator

n_clusters = 3
crypto_clusterizer = CryptoClusterizator(n_clusters)
crypto_clusterizer.clusterize('volatility', 'total_supply')
