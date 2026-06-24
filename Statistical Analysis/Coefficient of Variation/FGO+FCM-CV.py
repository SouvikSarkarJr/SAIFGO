import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import os

# ==========================================================
# 🔹 1. DATASET PATH
# ==========================================================
file_path = "/Users/souvik/Documents/Project-II/Data-sets/Delivery.csv"
n_clusters = 3
m = 2.0

# FGO parameters
n_fungi = 20
max_iter = 30
growth_rate = 0.5
spore_rate = 0.3

# ==========================================================
# 🔹 2. LOAD DATA
# ==========================================================
if not os.path.exists(file_path):
    raise FileNotFoundError("Dataset not found.")

df = pd.read_csv(file_path)

# Select numeric columns
X = df.select_dtypes(include=[np.number])
X = X.drop(columns=["Id", "ID", "id"], errors="ignore")

if X.shape[1] == 0:
    raise ValueError("Dataset must contain numeric columns.")

# Scale data for clustering
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
# 🔹 4. FUNGAL GROWTH OPTIMIZATION (FGO)
# ==========================================================

fungi = np.random.uniform(
    low=X_scaled.min(axis=0),
    high=X_scaled.max(axis=0),
    size=(n_fungi, n_clusters, n_features)
)

fitness_scores = np.array([fitness(f) for f in fungi])
best_idx = np.argmin(fitness_scores)
global_best = fungi[best_idx].copy()

for iteration in range(max_iter):

    for i in range(n_fungi):

        growth = growth_rate * np.random.rand() * (global_best - fungi[i])
        spore = spore_rate * np.random.randn(n_clusters, n_features)

        new_position = fungi[i] + growth + spore

        new_position = np.clip(
            new_position,
            X_scaled.min(axis=0),
            X_scaled.max(axis=0)
        )

        new_score = fitness(new_position)

        if new_score < fitness_scores[i]:
            fungi[i] = new_position
            fitness_scores[i] = new_score

    best_idx = np.argmin(fitness_scores)
    global_best = fungi[best_idx].copy()

    print(f"Iteration {iteration+1}/{max_iter} - Best XB: {fitness_scores[best_idx]:.6f}")

# ==========================================================
# 🔹 5. FINAL FCM USING FGO CENTROIDS
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

print("\n============= FGO + FCM CLUSTERING DONE =============")
print(pd.Series(labels).value_counts())
print("====================================================\n")

# ==========================================================
# 🔹 6. COEFFICIENT OF VARIATION (USING ORIGINAL DATA)
# ==========================================================

# Use ORIGINAL data (not scaled)
X_original = X.copy()
X_original["Cluster"] = labels

feature_name = X_original.columns[0]   # First feature
cv_values = []

for c in range(n_clusters):
    cluster_data = X_original[X_original["Cluster"] == c][feature_name]

    mean = np.mean(cluster_data)
    std = np.std(cluster_data, ddof=1)

    if abs(mean) > 1e-6:
        cv = std / mean
    else:
        cv = 0

    cv_values.append(cv)

print("CV Values:", cv_values)

# ==========================================================
# 🔹 7. PLOT CV GRAPH
# ==========================================================

plt.figure()
plt.bar(range(n_clusters), cv_values)
plt.xlabel("Clusters")
plt.ylabel("Coefficient of Variation (CV)")
plt.title(f"Coefficient of Variation - {feature_name} (FGO + FCM) Delivery Dataset")
plt.xticks(range(n_clusters), [f"C{c}" for c in range(n_clusters)])
plt.tight_layout()

output_file = "FGO+FCM.jpg"
plt.savefig(output_file, format="jpg")
plt.close()

print(f"CV plot saved as: {output_file}")