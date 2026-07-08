# ==========================================================
# PSO + K-MEANS + COEFFICIENT OF VARIATION
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
file_path = "#"  # Change
n_clusters = 3

# PSO parameters
n_particles = 20
max_iter = 30
w = 0.7
c1 = 1.5
c2 = 1.5

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
# 🔹 5. PARTICLE SWARM OPTIMIZATION
# ==========================================================

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

for iteration in range(max_iter):

    for i in range(n_particles):

        r1 = np.random.rand()
        r2 = np.random.rand()

        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (personal_best[i] - particles[i])
            + c2 * r2 * (global_best - particles[i])
        )

        particles[i] += velocities[i]

        score = fitness(particles[i])

        if score < personal_best_scores[i]:
            personal_best[i] = particles[i].copy()
            personal_best_scores[i] = score

    global_best_index = np.argmin(personal_best_scores)
    global_best = personal_best[global_best_index].copy()

    print(f"Iteration {iteration+1}/{max_iter} - Best SSE: {personal_best_scores[global_best_index]:.6f}")

# ==========================================================
# 🔹 6. FINAL K-MEANS USING PSO CENTROIDS
# ==========================================================

final_centroids, labels = kmeans_from_centroids(X_scaled, global_best)

# Add cluster labels to original data
X_original = pd.DataFrame(X, columns=X.columns)
X_original["Cluster"] = labels

print("\n================ PSO + K-MEANS DONE ================")
print(pd.Series(labels).value_counts())
print("====================================================\n")

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
plt.title(f"Coefficient of Variation - {feature_name} (PSO + K-Means) # Dataset") #Chnage
plt.xticks(range(n_clusters), [f"C{c}" for c in range(n_clusters)])
plt.tight_layout()

output_file = "PSO+KMeans.jpg"  # Change
plt.savefig(output_file, format="jpg")
plt.close()

print(f"CV plot saved as: {output_file}")
