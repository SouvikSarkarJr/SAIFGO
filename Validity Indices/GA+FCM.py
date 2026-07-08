# ==========================================================
# GENETIC ALGORITHM (GA) + FUZZY C-MEANS (FCM)
# + 5 VALIDITY INDICES
# Fully Corrected Version (No Shape Errors)
# ==========================================================

import numpy as np
import pandas as pd
import skfuzzy as fuzz
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
# 🔹 1. SETTINGS (EDIT HERE)
# ==========================================================
file_path = "/Users/souviksarkarjr./Downloads/Datasets/Bank.csv"
n_clusters = 3
m = 2.0

pop_size = 20
generations = 30
mutation_rate = 0.1

# ==========================================================
# 🔹 2. LOAD DATA
# ==========================================================
df = pd.read_csv(file_path)

# Keep only numeric columns
X = df.select_dtypes(include=[np.number]).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_samples, n_features = X_scaled.shape

# ==========================================================
# 🔹 3. HELPER FUNCTIONS
# ==========================================================

# --- Compute membership from centroids ---
def compute_membership(X, centroids, m):
    dist = cdist(X, centroids, metric='euclidean')
    dist = np.fmax(dist, np.finfo(np.float64).eps)

    power = 2 / (m - 1)
    temp = dist ** power
    denominator = temp[:, :, None] / temp[:, None, :]
    u = 1 / np.sum(denominator, axis=2)

    return u.T  # shape (c, n_samples)


# --- Xie-Beni Index ---
def xie_beni_index(X, u, centroids, m):
    n = X.shape[0]
    dist = cdist(X, centroids, metric='euclidean') ** 2
    numerator = np.sum((u.T ** m) * dist)

    centroid_distances = cdist(centroids, centroids)
    np.fill_diagonal(centroid_distances, np.inf)
    min_centroid_dist = np.min(centroid_distances) ** 2

    return numerator / (n * min_centroid_dist)


# --- Dunn Index ---
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
# 🔹 4. FITNESS FUNCTION
# ==========================================================
def fitness(centroids):
    u0 = compute_membership(X_scaled, centroids, m)

    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        X_scaled.T,
        c=n_clusters,
        m=m,
        error=0.005,
        maxiter=300,
        init=u0,
        seed=42
    )

    return xie_beni_index(X_scaled, u, cntr, m)


# ==========================================================
# 🔹 5. GENETIC ALGORITHM
# ==========================================================

# Initialize population
def initialize_population():
    population = []
    for _ in range(pop_size):
        centroids = np.random.uniform(
            low=X_scaled.min(axis=0),
            high=X_scaled.max(axis=0),
            size=(n_clusters, n_features)
        )
        population.append(centroids)
    return population


def selection(population, scores):
    idx = np.argsort(scores)
    return [population[i] for i in idx[:pop_size // 2]]


def crossover(parent1, parent2):
    alpha = np.random.rand()
    return alpha * parent1 + (1 - alpha) * parent2


def mutation(centroids):
    if random.random() < mutation_rate:
        noise = np.random.normal(0, 0.1, centroids.shape)
        centroids += noise
    return centroids


# ==========================================================
# 🔹 6. RUN GA
# ==========================================================
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

    print(f"Generation {gen+1}/{generations} - Best XB: {min(scores):.6f}")

# Best solution
final_scores = [fitness(ind) for ind in population]
best_index = np.argmin(final_scores)
best_centroids = population[best_index]

# ==========================================================
# 🔹 7. FINAL FCM USING GA-OPTIMIZED CENTROIDS
# ==========================================================
u0 = compute_membership(X_scaled, best_centroids, m)

cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    X_scaled.T,
    c=n_clusters,
    m=m,
    error=0.005,
    maxiter=1000,
    init=u0,
    seed=42
)

labels = np.argmax(u, axis=0)

# ==========================================================
# 🔹 8. VALIDITY INDICES
# ==========================================================
s_score = silhouette_score(X_scaled, labels)
ch_index = calinski_harabasz_score(X_scaled, labels)
db_index = davies_bouldin_score(X_scaled, labels)
xb_score = xie_beni_index(X_scaled, u, cntr, m)
dunn = dunn_index(X_scaled, labels)

# Calculate total time taken
end_time = time.time()
execution_time = end_time - start_time

# ==========================================================
# 🔹 9. PRINT RESULTS
# ==========================================================
print("\n============= GA + FCM CLUSTERING RESULTS =============")
print(f"Clusters: {n_clusters}")
print("-------------------------------------------------------")
print(f"Silhouette Score:        {s_score:.6f} (Higher=Better)")
print(f"Xie-Beni Index:          {xb_score:.6f} (Lower=Better)")
print(f"Calinski-Harabasz Index: {ch_index:.6f} (Higher=Better)")
print(f"Dunn Index:              {dunn:.6f} (Higher=Better)")
print(f"Davies-Bouldin Index:    {db_index:.6f} (Lower=Better)")
print(f"Execution Time:          {execution_time:.4f} seconds")  # <--- Added output
print("=======================================================\n")