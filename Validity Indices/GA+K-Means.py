# ==========================================================
# GENETIC ALGORITHM (GA) + K-MEANS
# + SAME 5 VALIDITY INDICES
# ==========================================================

import numpy as np
import pandas as pd
import random
import time  # <--- Added for execution time tracking
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from scipy.spatial.distance import cdist
from itertools import combinations

# Start the execution timer
start_time = time.time()

# ==========================================================
# 🔹 1. SETTINGS
# ==========================================================
file_path = "/Users/souviksarkarjr./Downloads/Bank.csv"
n_clusters = 3

pop_size = 20
generations = 30
mutation_rate = 0.1

# ==========================================================
# 🔹 2. LOAD DATA
# ==========================================================
df = pd.read_csv(file_path)

X = df.select_dtypes(include=[np.number]).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_samples, n_features = X_scaled.shape

# ==========================================================
# 🔹 3. MANUAL K-MEANS
# ==========================================================

def kmeans_from_centroids(X, centroids, max_iter=100):
    for _ in range(max_iter):
        distances = cdist(X, centroids)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[labels == k].mean(axis=0) if np.any(labels == k)
            else centroids[k]
            for k in range(n_clusters)
        ])

        if np.allclose(new_centroids, centroids):
            break

        centroids = new_centroids

    return centroids, labels

# ==========================================================
# 🔹 4. FITNESS FUNCTION (SSE)
# ==========================================================

def fitness(centroids):
    final_centroids, labels = kmeans_from_centroids(X_scaled, centroids)
    distances = cdist(X_scaled, final_centroids)
    sse = np.sum((distances[np.arange(n_samples), labels]) ** 2)
    return sse  # minimize

# ==========================================================
# 🔹 5. XIE-BENI INDEX (Hard version)
# ==========================================================

def xie_beni_index(X, centroids, labels):
    n = X.shape[0]
    distances = cdist(X, centroids) ** 2
    numerator = np.sum(distances[np.arange(n), labels])

    centroid_distances = cdist(centroids, centroids)
    np.fill_diagonal(centroid_distances, np.inf)
    min_centroid_dist = np.min(centroid_distances) ** 2

    return numerator / (n * min_centroid_dist)

# ==========================================================
# 🔹 6. DUNN INDEX
# ==========================================================

def dunn_index(X, labels):
    clusters = np.unique(labels)
    min_inter = np.inf

    for (i, j) in combinations(clusters, 2):
        dist = cdist(X[labels == i], X[labels == j])
        min_inter = min(min_inter, np.min(dist))

    max_intra = 0
    for i in clusters:
        points = X[labels == i]
        if len(points) > 1:
            dist = cdist(points, points)
            max_intra = max(max_intra, np.max(dist))

    return min_inter / max_intra

# ==========================================================
# 🔹 7. GENETIC ALGORITHM
# ==========================================================

def initialize_population():
    return [
        np.random.uniform(
            low=X_scaled.min(axis=0),
            high=X_scaled.max(axis=0),
            size=(n_clusters, n_features)
        )
        for _ in range(pop_size)
    ]

def selection(population, scores):
    idx = np.argsort(scores)
    return [population[i] for i in idx[:pop_size // 2]]

def crossover(p1, p2):
    alpha = np.random.rand()
    return alpha * p1 + (1 - alpha) * p2

def mutation(centroids):
    if random.random() < mutation_rate:
        centroids += np.random.normal(0, 0.1, centroids.shape)
    return centroids

population = initialize_population()

for gen in range(generations):
    scores = [fitness(ind) for ind in population]
    selected = selection(population, scores)

    new_population = selected.copy()

    while len(new_population) < pop_size:
        p1, p2 = random.sample(selected, 2)
        child = crossover(p1, p2)
        child = mutation(child)
        new_population.append(child)

    population = new_population
    print(f"Generation {gen+1}/{generations} - Best SSE: {min(scores):.6f}")

# Best solution
final_scores = [fitness(ind) for ind in population]
best_centroids = population[np.argmin(final_scores)]

# ==========================================================
# 🔹 8. FINAL K-MEANS USING GA CENTROIDS
# ==========================================================

final_centroids, labels = kmeans_from_centroids(X_scaled, best_centroids)

# ==========================================================
# 🔹 9. VALIDITY INDICES (UNCHANGED)
# ==========================================================

s_score = silhouette_score(X_scaled, labels)
ch_index = calinski_harabasz_score(X_scaled, labels)
db_index = davies_bouldin_score(X_scaled, labels)
xb_score = xie_beni_index(X_scaled, final_centroids, labels)
dunn = dunn_index(X_scaled, labels)

# Calculate total time taken
end_time = time.time()
execution_time = end_time - start_time

# ==========================================================
# 🔹 10. PRINT RESULTS
# ==========================================================

print("\n============= GA + K-MEANS CLUSTERING RESULTS =============")
print(f"Clusters: {n_clusters}")
print("-----------------------------------------------------------")
print(f"Silhouette Score:        {s_score:.6f} (Higher=Better)")
print(f"Xie-Beni Index:          {xb_score:.6f} (Lower=Better)")
print(f"Calinski-Harabasz Index: {ch_index:.6f} (Higher=Better)")
print(f"Dunn Index:              {dunn:.6f} (Higher=Better)")
print(f"Davies-Bouldin Index:    {db_index:.6f} (Lower=Better)")
print(f"Execution Time:          {execution_time:.4f} seconds")  # <--- Added output
print("===========================================================\n")