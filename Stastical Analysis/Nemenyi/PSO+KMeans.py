# ==========================================================
# PSO + K-MEANS + KRUSKAL-WALLIS + NEMENYI
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from sklearn.preprocessing import StandardScaler
from scipy.stats import kruskal
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

clusters = np.unique(labels)
features = X.columns.tolist()

print("\n================ PSO + K-MEANS DONE ================")
print(pd.Series(labels).value_counts())
print("====================================================\n")

# ==========================================================
# 🔹 7. KRUSKAL + NEMENYI
# ==========================================================

significant_results = []

for i in range(len(features)):

    cluster_data = []
    for c in clusters:
        cluster_values = X_scaled[labels == c, i]
        cluster_data.append(cluster_values)

    stat, p_value = kruskal(*cluster_data)

    if p_value < 0.05:

        temp_df = pd.DataFrame({
            "values": np.concatenate(cluster_data),
            "groups": np.concatenate([
                np.repeat(f"C{c}", len(cluster_data[idx]))
                for idx, c in enumerate(clusters)
            ])
        })

        nemenyi = sp.posthoc_nemenyi(
            temp_df,
            val_col="values",
            group_col="groups"
        )

        significant_results.append((features[i], nemenyi))

# ==========================================================
# 🔹 8. GENERATE ONE COMBINED IMAGE
# ==========================================================

if len(significant_results) == 0:
    print("No significant features found.")
else:

    rows = len(significant_results)
    plt.figure(figsize=(6, 4 * rows))

    for idx, (feature_name, nemenyi_matrix) in enumerate(significant_results):
        plt.subplot(rows, 1, idx + 1)
        plt.imshow(nemenyi_matrix)
        plt.colorbar()
        plt.title(f"Nemenyi Post-Hoc (PSO + K-Means) # Dataset - {feature_name}") #Change
        plt.xticks(range(len(clusters)), [f"C{c}" for c in clusters])
        plt.yticks(range(len(clusters)), [f"C{c}" for c in clusters])

    plt.tight_layout()

    output_file = "PSO+KMeans.jpg"  # Change
    plt.savefig(output_file, format="jpg")
    plt.close()

    print(f"Single image saved as: {output_file}")
