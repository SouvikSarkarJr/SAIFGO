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
file_path = "/Users/souvik/Documents/Project-II/Data-sets/Wave.csv"
n_clusters = 3

# FGO parameters
n_agents = 20
max_iter = 30
growth_rate = 0.5
mutation_rate = 0.1

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

agents = np.random.uniform(
    low=X_scaled.min(axis=0),
    high=X_scaled.max(axis=0),
    size=(n_agents, n_clusters, n_features)
)

fitness_scores = np.array([fitness(a) for a in agents])
best_idx = np.argmin(fitness_scores)
global_best = agents[best_idx].copy()

for iteration in range(max_iter):

    for i in range(n_agents):

        # Growth toward best solution
        growth = growth_rate * np.random.rand(n_clusters, n_features) * \
                 (global_best - agents[i])

        # Random local mutation (spore variation)
        mutation = mutation_rate * np.random.randn(n_clusters, n_features)

        new_position = agents[i] + growth + mutation

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

    print(f"Iteration {iteration+1}/{max_iter} - Best SSE: {fitness_scores[best_idx]:.6f}")

# ==========================================================
# 🔹 6. FINAL K-MEANS USING FGO CENTROIDS
# ==========================================================

final_centroids, labels = kmeans_from_centroids(X_scaled, global_best)

clusters = np.unique(labels)
features = X.columns.tolist()

print("\n================ FGO + K-MEANS DONE ================")
print(pd.Series(labels).value_counts())
print("=====================================================\n")

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
        plt.title(f"Nemenyi Post-Hoc Wave Dataset - {feature_name}")
        plt.xticks(range(len(clusters)), [f"C{c}" for c in clusters])
        plt.yticks(range(len(clusters)), [f"C{c}" for c in clusters])

    plt.tight_layout()

    output_file = "FGO+KMeans_Wave.jpg"
    plt.savefig(output_file, format="jpg")
    plt.close()

    print(f"Single image saved as: {output_file}")