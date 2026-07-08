# ==========================================================
# ISGO (Improved Social Group Optimization) + FCM + BOXPLOT
# ==========================================================

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
file_path = "#" # Change
n_clusters = 3
m = 2.0

# ISGO parameters
n_agents = 20
max_iter = 30
alpha = 0.5
beta = 0.3

# ==========================================================
# 🔹 2. LOAD DATA
# ==========================================================
if not os.path.exists(file_path):
    raise FileNotFoundError("Dataset not found.")

df = pd.read_csv(file_path)

X = df.select_dtypes(include=[np.number])
X = X.drop(columns=["Id", "ID", "id"], errors="ignore")

if X.shape[1] == 0:
    raise ValueError("Dataset must contain numeric columns.")

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
# 🔹 4. ISGO OPTIMIZATION
# ==========================================================

agents = np.random.uniform(
    low=X_scaled.min(axis=0),
    high=X_scaled.max(axis=0),
    size=(n_agents, n_clusters, n_features)
)

fitness_scores = np.array([fitness(a) for a in agents])
best_idx = np.argmin(fitness_scores)
global_best = agents[best_idx].copy()

for iteration in range(max_iter):

    group_mean = np.mean(agents, axis=0)

    for i in range(n_agents):

        r1 = np.random.rand()
        r2 = np.random.rand()

        new_position = (
            agents[i]
            + alpha * r1 * (global_best - agents[i])
            + beta * r2 * (group_mean - agents[i])
        )

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
    global_best = agents[best_idx].copy()

    print(f"Iteration {iteration+1}/{max_iter} - Best XB: {fitness_scores[best_idx]:.6f}")

# ==========================================================
# 🔹 5. FINAL FCM USING ISGO CENTROIDS
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

# Add cluster labels to original data
X_original = pd.DataFrame(X, columns=X.columns)
X_original["Cluster"] = labels

print("\n============= ISGO + FCM CLUSTERING DONE =============")
print(pd.Series(labels).value_counts())
print("=======================================================\n")

# ==========================================================
# 🔹 6. GENERATE SINGLE BOXPLOT IMAGE
# ==========================================================

plt.figure(figsize=(8, 6))

feature_name = X.columns[0]

data_to_plot = [
    X_original[X_original["Cluster"] == c][feature_name]
    for c in range(n_clusters)
]

plt.boxplot(data_to_plot)
plt.xlabel("Clusters")
plt.ylabel(feature_name)
plt.title(f"Boxplot of {feature_name} Across Clusters (ISGO + FCM) # Dataset") # Change
plt.xticks(range(1, n_clusters + 1), [f"C{c}" for c in range(n_clusters)])
plt.tight_layout()

output_file = "ISGO+FCM.jpg" # Change
plt.savefig(output_file, format="jpg")
plt.close()

print(f"Boxplot saved as: {output_file}")
