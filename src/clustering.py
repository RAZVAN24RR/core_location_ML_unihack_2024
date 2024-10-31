from sklearn.cluster import KMeans
import numpy as np

def group_parents(location, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(location)
    return kmeans.labels_, kmeans.cluster_centers_