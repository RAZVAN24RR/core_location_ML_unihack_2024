import numpy as np
from src.clustering import group_parents
from src.route_optimization import optimize_routes

#ex de date pentru loc parintilor

parent_locations = np.array([
    [45.7489, 21.2087],
    [45.7599, 21.2235],
    [45.7435, 21.2242],
])

#cluesterin
groups, centroids = group_parents(parent_locations)
print("Grupuri de parinti:", groups)
print("Centroide", centroids)

#opt date
q_network, optimizer = optimize_routes(parent_locations)
print("Model de opt si opt date.")