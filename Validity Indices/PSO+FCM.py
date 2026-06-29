# ==========================================================
# PARTICLE SWARM OPTIMIZATION (PSO) + FUZZY C-MEANS (FCM)
# + 5 VALIDITY INDICES
# ==========================================================

import numpy as np
import pandas as pd
import skfuzzy as fuzz
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
m = 2.0

# PSO parameters
n_particles = 20
max_iter = 30
w = 0.7       # inertia weight
c1 = 1.5      # cognitive
c2 = 1.5      # social

# ==========================================================
# 🔹 2. LOAD DATA
# ==========================================================
df = pd.read_csv(file_path)

X = df.select_dtypes(include=[np.number]).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_samples, n_features = X_scaled.shape

# ==========================================================
# 🔹 3. HELPER FUNCTIONS
# ==========================================================

def compute_membership(X, centroids, m):
    dist = cdist(X, centroids)
    dist = np.fmax(dist, np.finfo(np.float64).eps)

    power = 2 / (m - 1)
    temp = dist ** power
    denominator = temp[:, :, None] / temp[:, None, :]
    u = 1 / np.sum(denominator, axis=2)

    return u.T


def xie_beni_index(X, u, centroids, m):
    n = X.shape[0]
    dist = cdist(X, centroids) ** 2
    numerator = np.sum((u.T ** m) * dist)

    centroid_distances = cdist(centroids, centroids)
    np.fill_diagonal(centroid_distances, np.inf)
    min_centroid_dist = np.min(centroid_distances) ** 2

    return numerator / (n * min_centroid_dist)


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
# 🔹 4. PARTICLE SWARM OPTIMIZATION
# ==========================================================

# Initialize particles
particles = np.random.uniform(
    low=X_scaled.min(axis=0),
    high=X_scaled.max(axis=0),
    size=(n_particles, n_clusters, n_features)
)

velocities = np.zeros_like(particles)

personal_best = particles.copy()
personal_best_scores = np.array([fitness(p) for p in particles])

global_best_index = np.argmin(personal_best_scores)
global_best = personal_best[global_best_index].copy()

# PSO loop
for iteration in range(max_iter):

    for i in range(n_particles):

        r1 = np.random.rand()
        r2 = np.random.rand()

        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (personal_best[i] - particles[i])
            + c2 * r2 * (global_best - particles[i])
        )

        particles[i] = particles[i] + velocities[i]

        score = fitness(particles[i])

        if score < personal_best_scores[i]:
            personal_best[i] = particles[i].copy()
            personal_best_scores[i] = score

    global_best_index = np.argmin(personal_best_scores)
    global_best = personal_best[global_best_index].copy()

    print(f"Iteration {iteration+1}/{max_iter} - Best XB: {personal_best_scores[global_best_index]:.6f}")

# ==========================================================
# 🔹 5. FINAL FCM USING PSO-OPTIMIZED CENTROIDS
# ==========================================================

u0 = compute_membership(X_scaled, global_best, m)

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
# 🔹 6. VALIDITY INDICES
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
# 🔹 7. PRINT RESULTS
# ==========================================================

print("\n============= PSO + FCM CLUSTERING RESULTS =============")
print(f"Clusters: {n_clusters}")
print("--------------------------------------------------------")
print(f"Silhouette Score:        {s_score:.6f} (Higher=Better)")
print(f"Xie-Beni Index:          {xb_score:.6f} (Lower=Better)")
print(f"Calinski-Harabasz Index: {ch_index:.6f} (Higher=Better)")
print(f"Dunn Index:              {dunn:.6f} (Higher=Better)")
print(f"Davies-Bouldin Index:    {db_index:.6f} (Lower=Better)")
print(f"Execution Time:          {execution_time:.4f} seconds")  # <--- Added output
print("========================================================\n")