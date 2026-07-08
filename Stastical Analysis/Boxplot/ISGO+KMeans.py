# ==========================================================
# ISGO + K-MEANS + BOXPLOT VISUALIZATION
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import os

# ==========================================================
# 🔹 1. DATASET PATH
# ==========================================================
file_path = "#" # Change
n_clusters = 3

# ISGO parameters
n_agents = 20
isgo_iterations = 30
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
# 🔹 5. IMPROVED SOCIAL GROUP OPTIMIZATION (ISGO)
# ==========================================================

agents = np.random.uniform(
    low=X_scaled.min(axis=0),
    high=X_scaled.max(axis=0),
    size=(n_agents, n_clusters, n_features)
)

fitness_scores = np.array([fitness(a) for a in agents])
best_idx = np.argmin(fitness_scores)
global_best = agents[best_idx].copy()

for iteration in range(isgo_iterations):

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

    print(f"Iteration {iteration+1}/{isgo_iterations} - Best SSE: {fitness_scores[best_idx]:.6f}")

# ==========================================================
# 🔹 6. FINAL K-MEANS USING ISGO CENTROIDS
# ==========================================================

final_centroids, labels = kmeans_from_centroids(X_scaled, global_best)

# Add cluster labels to original data
X_original = pd.DataFrame(X, columns=X.columns)
X_original["Cluster"] = labels

print("\n================ ISGO + K-MEANS DONE ================")
print(pd.Series(labels).value_counts())
print("=====================================================\n")

# ==========================================================
# 🔹 7. GENERATE SINGLE BOXPLOT IMAGE
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
plt.title(f"Boxplot of {feature_name} Across Clusters (ISGO + K-Means) # Dataset") # Chnage
plt.xticks(range(1, n_clusters + 1), [f"C{c}" for c in range(n_clusters)])
plt.tight_layout()

output_file = "ISGO+KMeans.jpg" # Chnage
plt.savefig(output_file, format="jpg")
plt.close()

print(f"Boxplot saved as: {output_file}")
