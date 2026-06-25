import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

from scipy.spatial.distance import cdist

# ==========================
# PARAMETERS
# ==========================
csv_path = "/Users/souviksarkarjr./Downloads/Experimental Dataset/Iris.csv"

n_clusters = 3
pop_size = 30
max_iter = 100

# Base parameters for global adaptive beta tuning
beta_min = 1.0
beta_max = 3.0

random.seed(42)
np.random.seed(42)

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv(csv_path)

if 'Id' in df.columns:
    df = df.drop(columns=['Id'])
elif 'id' in df.columns:
    df = df.drop(columns=['id'])

df = df.select_dtypes(include=[np.number])

if df.shape[1] == 0:
    raise ValueError("Dataset has no numeric columns!")

data = df.values

# ==========================
# NORMALIZATION
# ==========================
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# ==========================
# HELPER FUNCTIONS
# ==========================
def assign_clusters(data, centroids):
    distances = cdist(data, centroids, metric='euclidean')
    return np.argmin(distances, axis=1)


def clustering_fitness(data, centroids):
    labels = assign_clusters(data, centroids)
    fitness = 0
    for i in range(len(centroids)):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            fitness += np.sum((cluster_points - centroids[i]) ** 2)
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

    if intra_cluster == 0:
        return 0

    return inter_cluster / intra_cluster


def xie_beni_index(data, labels, centroids):
    N = len(data)
    K = len(centroids)
    numerator = 0

    for i in range(K):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            numerator += np.sum((cluster_points - centroids[i]) ** 2)

    centroid_dist = cdist(centroids, centroids)
    np.fill_diagonal(centroid_dist, np.inf)
    min_centroid_dist = np.min(centroid_dist)

    if min_centroid_dist == 0:
        return 0

    return numerator / (N * (min_centroid_dist ** 2))


# ==========================
# SEARCH SPACE
# ==========================
dim = data.shape[1]
lb = np.min(data, axis=0)
ub = np.max(data, axis=0)

# ==========================
# POPULATION INITIALIZATION
# ==========================
population = [
    np.random.uniform(lb, ub, (n_clusters, dim))
    for _ in range(pop_size)
]

# Calculate initial single-objective fitness scores
fitness = [
    clustering_fitness(data, p)
    for p in population
]

# Track global best solution
best_idx = np.argmin(fitness)
best_centroids = population[best_idx].copy()
best_score = fitness[best_idx]

convergence_curve = []

# ==========================
# FGO OPTIMIZATION LOOP
# ==========================
for t in range(max_iter):

    # --- Global Adaptive Step ---
    # Dynamically scale beta over time. It transitions smoothly from beta_max 
    # down to beta_min as iterations progress to tightly guide convergence.
    current_beta = beta_max - (beta_max - beta_min) * (t / max_iter)

    # Compute the time-dependent decreasing coefficient using the adaptive global beta
    R = (1 - t / max_iter) ** current_beta

    print(
        f"Iteration {t+1}/{max_iter} | "
        f"Beta: {current_beta:.4f} | "
        f"Best Fitness: {best_score:.6f}"
    )

    for i in range(pop_size):
        Xi = population[i].copy()

        # Select two distinct random vectors from the population
        j, k = np.random.choice(pop_size, 2, replace=False)

        # FGO exploitation/exploration formula
        new_solution = Xi + R * np.random.rand() * (population[j] - population[k])
        new_solution = np.clip(new_solution, lb, ub)

        # Evaluate the single-objective fitness of the candidate solution
        new_fitness = clustering_fitness(data, new_solution)

        # --- Greedy Selection Step ---
        if new_fitness < fitness[i]:
            population[i] = new_solution
            fitness[i] = new_fitness

        # Update the overall best found solution
        if fitness[i] < best_score:
            best_score = fitness[i]
            best_centroids = population[i].copy()

    convergence_curve.append(best_score)

# ==========================
# FINAL CLUSTERING
# ==========================
labels = assign_clusters(data, best_centroids)

# ==========================
# VALIDITY INDICES
# ==========================
s_score = silhouette_score(data, labels)
ch_score = calinski_harabasz_score(data, labels)
db_score = davies_bouldin_score(data, labels)
dunn = dunn_index(data, labels)
xb = xie_beni_index(data, labels, best_centroids)

# ==========================
# RESULTS
# ==========================
print("\n========== RESULTS (FGO + Single Objective Adaptive) ==========")
print("Best Fitness:", best_score)
print("Final Iteration Beta:", current_beta)
print("Silhouette Score (S):", s_score)
print("Calinski-Harabasz (CH):", ch_score)
print("Dunn Index:", dunn)
print("Davies-Bouldin (DB):", db_score)
print("Xie-Beni (XB):", xb)