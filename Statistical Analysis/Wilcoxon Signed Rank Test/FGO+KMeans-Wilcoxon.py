import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon
from scipy.spatial.distance import cdist
import os

# ==========================================================
# 🔹 1. SETTINGS
# ==========================================================
file_path = "/Users/souvik/Documents/Project-II/Data-sets/Iris.csv" # Change
n_clusters = 3

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

X = df.select_dtypes(include=[np.number])

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

for iteration in range(max_iter):

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

    print(f"Iteration {iteration+1}/{max_iter} - Best SSE: {fitness_scores[best_idx]:.6f}")

# ==========================================================
# 🔹 6. FINAL K-MEANS USING FGO CENTROIDS
# ==========================================================

final_centroids, labels = kmeans_from_centroids(X_scaled, global_best)
df["Cluster"] = labels

print("\n================ FGO + K-MEANS DONE ================")
print(df["Cluster"].value_counts())
print("====================================================\n")

# ==========================================================
# 🔹 7. WILCOXON SIGNED-RANK TEST
# ==========================================================

clusters = np.unique(labels)

if len(clusters) < 2:
    raise ValueError("Need at least 2 clusters for Wilcoxon Test.")

cluster_0 = X_scaled[labels == clusters[0]]
cluster_1 = X_scaled[labels == clusters[1]]

min_size = min(len(cluster_0), len(cluster_1))
cluster_0 = cluster_0[:min_size]
cluster_1 = cluster_1[:min_size]

features = X.columns.tolist()
p_values = []

print("============= WILCOXON TEST RESULTS =============")

for i in range(len(features)):
    stat, p_value = wilcoxon(cluster_0[:, i], cluster_1[:, i])
    p_values.append(p_value)

    print(f"\nFeature: {features[i]}")
    print(f"Statistic = {stat:.4f}")
    print(f"P-value   = {p_value:.6f}")

    if p_value < 0.05:
        print("Significant Difference")
    else:
        print("Not Significant")

print("==================================================\n")

# ==========================================================
# 🔹 8. GENERATE GRAPH
# ==========================================================

plt.figure()
plt.bar(features, p_values)
plt.axhline(y=0.05)
plt.xticks(rotation=45)
plt.title("Wilcoxon Signed-Rank Test (FGO + K-Means)") 
plt.xlabel("Features (Iris Dataset)") # Change
plt.ylabel("P-values")
plt.tight_layout()

output_file = "Iris_FGO_KMeans_Wilcoxon.jpg" # Change
plt.savefig(output_file, format="jpg")
plt.close()

print(f"Graph saved successfully as: {output_file}")