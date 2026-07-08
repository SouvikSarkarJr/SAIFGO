# ==========================================================
# GENERAL TREND OPTIMIZER (GT) + K-MEANS
# + SAME 5 VALIDITY INDICES
# ==========================================================

import numpy as np
import pandas as pd
import time
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
file_path = "/Users/souviksarkarjr./Downloads/Datasets/Bank.csv"
n_clusters = 3

# GT parameters
n_agents = 20
max_iter = 30
w_trend = 0.4   # Weight assigned to personal trend memory
w_global = 0.5  # Weight assigned to population trend momentum

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
    return sse

# ==========================================================
# 🔹 5. XIE-BENI INDEX
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
# 🔹 7. GENERAL TREND OPTIMIZER (GT)
# ==========================================================

agents = np.random.uniform(
    low=X_scaled.min(axis=0),
    high=X_scaled.max(axis=0),
    size=(n_agents, n_clusters, n_features)
)

trends = np.zeros_like(agents)

fitness_scores = np.array([fitness(a) for a in agents])
best_idx = np.argmin(fitness_scores)
global_best = agents[best_idx].copy()

for iteration in range(max_iter):

    global_mean_position = np.mean(agents, axis=0)

    for i in range(n_agents):

        r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
        exploration_noise = np.random.randn(n_clusters, n_features) * 0.02
        
        global_trend_vector = w_global * r1 * (global_best - global_mean_position)
        new_position = agents[i] + (w_trend * r2 * trends[i]) + global_trend_vector + r3 * exploration_noise

        new_position = np.clip(
            new_position,
            X_scaled.min(axis=0),
            X_scaled.max(axis=0)
        )

        new_score = fitness(new_position)

        if new_score < fitness_scores[i]:
            trends[i] = new_position - agents[i]
            agents[i] = new_position
            fitness_scores[i] = new_score
        else:
            trends[i] *= 0.5

    best_idx = np.argmin(fitness_scores)
    global_best = agents[best_idx].copy()

    print(f"Iteration {iteration+1}/{max_iter} - Best SSE: {fitness_scores[best_idx]:.6f}")

# ==========================================================
# 🔹 8. FINAL K-MEANS USING GT CENTROIDS
# ==========================================================

final_centroids, labels = kmeans_from_centroids(X_scaled, global_best)

# ==========================================================
# 🔹 9. VALIDITY INDICES
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

print("\n============= GT + K-MEANS CLUSTERING RESULTS =============")
print(f"Clusters: {n_clusters}")
print("-----------------------------------------------------------")
print(f"Silhouette Score:        {s_score:.6f} (Higher=Better)")
print(f"Xie-Beni Index:          {xb_score:.6f} (Lower=Better)")
print(f"Calinski-Harabasz Index: {ch_index:.6f} (Higher=Better)")
print(f"Dunn Index:              {dunn:.6f} (Higher=Better)")
print(f"Davies-Bouldin Index:    {db_index:.6f} (Lower=Better)")
print(f"Execution Time:          {execution_time:.4f} seconds")
print("===========================================================\n")