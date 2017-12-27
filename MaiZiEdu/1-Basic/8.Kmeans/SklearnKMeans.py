#encoding=utf-8
import warnings
import pylab as pl
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs

# #############################################################################
# Generate sample data
np.random.seed(0)
k=3

X=np.array([[1, 1],[1, 1.5],[1.5, 1.5],[2, 1],[4.5, 3],[5, 3],[5, 3.5]])

# #############################################################################
# Compute clustering with Means
k_means = KMeans(init='k-means++', n_clusters=k, n_init=10)

k_means.fit(X)

k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
print (k_means_cluster_centers)
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
print (k_means_labels)
pl.scatter(X[:, 0], X[:, 1])
pl.show()


