import numpy as np
import pandas as pd
import math
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist

csv_path = "/Users/souviksarkarjr./Downloads/Experimental Dataset/Iris.csv"   # CSV File Path
n_clusters = 3
pop_size = 30
max_iter = 100
beta = 2
random.seed(42)
np.random.seed(42)

df = pd.read_csv(csv_path)

# Drop ID column if present 
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])
elif 'id' in df.columns:
    df = df.drop(columns=['id'])

df = df.select_dtypes(include=[np.number])
if df.shape[1] == 0:
    raise ValueError("Dataset has no numeric columns!")

data = df.values

# Normalizing Data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# HELPER FUNCTIONS
def assign_clusters(data, centroids):
    distances = cdist(data, centroids, metric='euclidean')
    return np.argmin(distances, axis=1)

def clustering_fitness(data, centroids):
    labels = assign_clusters(data, centroids)
    fitness = 0
    for i in range(len(centroids)):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            fitness += np.sum((cluster_points - centroids[i])**2)
    return fitness

def dunn_index(data, labels):
    unique_clusters = np.unique(labels)
    if len(unique_clusters) < 2:
        return 0
    clusters = [data[labels == k] for k in unique_clusters]

    inter_cluster = np.inf
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            dist = np.min(cdist(clusters[i], clusters[j]))
            inter_cluster = min(inter_cluster, dist)

    intra_cluster = 0
    for cluster in clusters:
        if len(cluster) > 1:
            dist = np.max(cdist(cluster, cluster))
            intra_cluster = max(intra_cluster, dist)

    return inter_cluster / intra_cluster if intra_cluster != 0 else 0

def xie_beni_index(data, labels, centroids):
    N = len(data)
    K = len(centroids)

    numerator = 0
    for i in range(K):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            numerator += np.sum((cluster_points - centroids[i])**2)

    centroid_dist = cdist(centroids, centroids)
    np.fill_diagonal(centroid_dist, np.inf)
    min_centroid_dist = np.min(centroid_dist)

    return numerator / (N * (min_centroid_dist ** 2)) if min_centroid_dist != 0 else 0

# --- ABLATION CONFIGURATION: STANDARD FGO + OBL ONLY ---
dim = data.shape[1]
lb = np.min(data, axis=0)
ub = np.max(data, axis=0)

# 1. Keep OBL Initialization
extended_population = []
for _ in range(pop_size):
    centroids = np.random.uniform(lb, ub, (n_clusters, dim))
    opposite = lb + ub - centroids
    extended_population.append(centroids)
    extended_population.append(opposite)

extended_fitness = [clustering_fitness(data, p) for p in extended_population]
sorted_indices = np.argsort(extended_fitness)

population = [extended_population[idx] for idx in sorted_indices[:pop_size]]
fitness = [extended_fitness[idx] for idx in sorted_indices[:pop_size]]

best_idx = np.argmin(fitness)
best_centroids = population[best_idx].copy()
best_score = fitness[best_idx]

convergence_curve = []

# OPTIMIZATION LOOP
for t in range(max_iter):
    # Standard time-varying mechanism remaining
    R = (1 - t / max_iter) ** beta
    
    print(f"Iteration {t+1}/{max_iter} | Best Fitness: {best_score:.6f}")

    for i in range(pop_size):
        Xi = population[i].copy()

        # 2. Reverted to Standard FGO Core Operator 
        # (Standard differential exploration step without strategy selection or Levy flights)
        j, k = np.random.choice(pop_size, 2, replace=False)
        new_solution = Xi + R * np.random.rand() * (population[j] - population[k])

        # Boundary control
        new_solution = np.clip(new_solution, lb, ub)
        new_fit = clustering_fitness(data, new_solution)

        # Standard greedy selection
        if new_fit < fitness[i]:
            population[i] = new_solution
            fitness[i] = new_fit

            if new_fit < best_score:
                best_score = new_fit
                best_centroids = new_solution.copy()

    convergence_curve.append(best_score)

# FINAL CLUSTER ASSIGNMENT
labels = assign_clusters(data, best_centroids)

# VALIDITY INDICES
s_score = silhouette_score(data, labels)
ch_score = calinski_harabasz_score(data, labels)
db_score = davies_bouldin_score(data, labels)
dunn = dunn_index(data, labels)
xb = xie_beni_index(data, labels, best_centroids)

print("\n========== ABLATION STUDY RESULTS (Standard-FGO + OBL) ==========")
print("Best Fitness:", best_score)
print("Silhouette Score (S):", s_score)
print("Calinski-Harabasz (CH):", ch_score)
print("Dunn Index:", dunn)
print("Davies-Bouldin (DB):", db_score)
print("Xie-Beni (XB):", xb)