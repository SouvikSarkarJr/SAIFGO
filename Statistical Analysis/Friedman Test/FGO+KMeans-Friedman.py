import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import friedmanchisquare
from scipy.spatial.distance import cdist
import os

# ==========================================================
# 🔹 1. DATASET PATH
# ==========================================================
file_path = "/Users/souvik/Documents/Project-II/Data-sets/Bank.csv" # Change
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
# 🔹 7. FRIEDMAN TEST
# ==========================================================

clusters = np.unique(labels)

if len(clusters) < 3:
    raise ValueError("Friedman Test requires at least 3 clusters.")

features = X.columns.tolist()
p_values = []

print("============= FRIEDMAN TEST RESULTS =============")

for i in range(len(features)):

    cluster_data = []

    for c in clusters:
        cluster_values = X_scaled[labels == c, i]
        cluster_data.append(cluster_values)

    min_size = min(len(arr) for arr in cluster_data)
    cluster_data = [arr[:min_size] for arr in cluster_data]

    stat, p_value = friedmanchisquare(*cluster_data)
    p_values.append(p_value)

    print(f"\nFeature: {features[i]}")
    print(f"Friedman Statistic = {stat:.4f}")
    print(f"P-value = {p_value:.6f}")

    if p_value < 0.05:
        print("Significant difference among clusters")
    else:
        print("No significant difference")

print("=================================================\n")

# ==========================================================
# 🔹 8. GENERATE GRAPH
# ==========================================================

plt.figure()
plt.bar(features, p_values)
plt.axhline(y=0.05)
plt.xticks(rotation=45)
plt.title("Friedman Test (FGO + K-Means)")
plt.xlabel("Features (Bank Dataset)") # Change
plt.ylabel("P-values")
plt.tight_layout()

output_file = "Bank_FGO_KMeans_Friedman.jpg" # Change
plt.savefig(output_file, format="jpg")
plt.close()

print(f"Graph saved successfully as: {output_file}")