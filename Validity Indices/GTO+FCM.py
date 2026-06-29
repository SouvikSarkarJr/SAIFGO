# ==========================================================
# GORILLA TROOP OPTIMIZATION (GTO) + FUZZY C-MEANS (FCM)
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

# GTO parameters
n_agents = 20
max_iter = 30
lb = -1
ub = 1

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
# 🔹 4. GORILLA TROOP OPTIMIZATION (GTO)
# ==========================================================

# Initialize gorillas (agents)
agents = np.random.uniform(
    low=X_scaled.min(axis=0),
    high=X_scaled.max(axis=0),
    size=(n_agents, n_clusters, n_features)
)

fitness_scores = np.array([fitness(a) for a in agents])

best_idx = np.argmin(fitness_scores)
silverback = agents[best_idx].copy()

for iteration in range(max_iter):

    for i in range(n_agents):

        r = np.random.rand()
        C = 2 * r - 1
        F = np.random.rand()

        if F < 0.5:
            # Exploration phase
            new_position = agents[i] + C * (agents[i] - silverback)
        else:
            # Exploitation phase
            new_position = silverback - C * np.abs(silverback - agents[i])

        # Boundary control
        new_position = np.clip(
            new_position,
            X_scaled.min(axis=0),
            X_scaled.max(axis=0)
        )

        new_score = fitness(new_position)

        if new_score < fitness_scores[i]:
            agents[i] = new_position
            fitness_scores[i] = new_score

    best_idx = np.argmin(fitness_scores)
    silverback = agents[best_idx].copy()

    print(f"Iteration {iteration+1}/{max_iter} - Best XB: {fitness_scores[best_idx]:.6f}")

# ==========================================================
# 🔹 5. FINAL FCM USING GTO-OPTIMIZED CENTROIDS
# ==========================================================

u0 = compute_membership(X_scaled, silverback, m)

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

print("\n============= GTO + FCM CLUSTERING RESULTS =============")
print(f"Clusters: {n_clusters}")
print("--------------------------------------------------------")
print(f"Silhouette Score:        {s_score:.6f} (Higher=Better)")
print(f"Xie-Beni Index:          {xb_score:.6f} (Lower=Better)")
print(f"Calinski-Harabasz Index: {ch_index:.6f} (Higher=Better)")
print(f"Dunn Index:              {dunn:.6f} (Higher=Better)")
print(f"Davies-Bouldin Index:    {db_index:.6f} (Lower=Better)")
print(f"Execution Time:          {execution_time:.4f} seconds")  # <--- Added output
print("========================================================\n")