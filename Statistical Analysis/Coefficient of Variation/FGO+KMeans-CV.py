import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import os

# ==========================================================
# 🔹 1. DATASET PATH
# ==========================================================
file_path = "/Users/souvik/Documents/Project-II/Data-sets/Delivery.csv" # Change
n_clusters = 3

# FGO parameters
n_fungi = 20
fgo_iterations = 30
growth_rate = 0.5
spore_rate = 0.3

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
# 🔹 5. FUNGAL GROWTH OPTIMIZATION (FGO)
# ==========================================================

fungi = np.random.uniform(
    low=X_scaled.min(axis=0),
    high=X_scaled.max(axis=0),
    size=(n_fungi, n_clusters, n_features)
)

fitness_scores = np.array([fitness(f) for f in fungi])
best_idx = np.argmin(fitness_scores)
global_best = fungi[best_idx].copy()

for iteration in range(fgo_iterations):

    for i in range(n_fungi):

        # Mycelium growth toward best fungus
        growth = growth_rate * np.random.rand() * (global_best - fungi[i])

        # Spore mutation
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

    print(f"Iteration {iteration+1}/{fgo_iterations} - Best SSE: {fitness_scores[best_idx]:.6f}")

# ==========================================================
# 🔹 6. FINAL K-MEANS USING FGO CENTROIDS
# ==========================================================

final_centroids, labels = kmeans_from_centroids(X_scaled, global_best)

# Add cluster labels to original data
X_original = pd.DataFrame(X, columns=X.columns)
X_original["Cluster"] = labels

print("\n================ FGO + K-MEANS DONE ================")
print(pd.Series(labels).value_counts())
print("===================================================\n")

# ==========================================================
# 🔹 7. COEFFICIENT OF VARIATION (First Feature)
# ==========================================================

feature_name = X.columns[0]
cv_values = []

for c in range(n_clusters):
    cluster_data = X_original[X_original["Cluster"] == c][feature_name]
    mean = np.mean(cluster_data)
    std = np.std(cluster_data, ddof=1)
    cv = std / mean if mean != 0 else 0
    cv_values.append(cv)

# ==========================================================
# 🔹 8. PLOT SINGLE IMAGE
# ==========================================================

plt.figure()
plt.bar(range(n_clusters), cv_values)
plt.xlabel("Clusters")
plt.ylabel("Coefficient of Variation (CV)")
plt.title(f"Coefficient of Variation - {feature_name} (FGO + K-Means) Delivery Dataset") # Change
plt.xticks(range(n_clusters), [f"C{c}" for c in range(n_clusters)])
plt.tight_layout()

output_file = "FGO+KMeans.jpg" # Change
plt.savefig(output_file, format="jpg")
plt.close()

print(f"CV plot saved as: {output_file}")