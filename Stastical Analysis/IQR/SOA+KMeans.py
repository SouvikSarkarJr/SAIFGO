# ==========================================================
# SOA (Secant Optimization Algorithm) + K-MEANS
# + KRUSKAL-WALLIS + MEDIAN & IQR
# (ONE IMAGE OUTPUT)
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import kruskal
from scipy.spatial.distance import cdist
import os

# ==========================================================
# 🔹 1. DATASET PATH
# ==========================================================
file_path = "#" # Change
n_clusters = 3

# SOA parameters
n_agents = 20
max_iter = 30

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
# 🔹 5. SECANT OPTIMIZATION ALGORITHM (SOA)
# ==========================================================

agents = np.random.uniform(
    low=X_scaled.min(axis=0),
    high=X_scaled.max(axis=0),
    size=(n_agents, n_clusters, n_features)
)

fitness_scores = np.array([fitness(a) for a in agents])
best_idx = np.argmin(fitness_scores)
global_best = agents[best_idx].copy()
global_best_score = fitness_scores[best_idx]

for iteration in range(max_iter):

    for i in range(n_agents):

        fitness_diff = agents[i] - global_best
        score_diff = fitness_scores[i] - global_best_score
        
        r1 = np.random.rand()
        r2 = np.random.rand()

        if abs(score_diff) > 1e-6:
            # Secant update calculation
            secant_step = fitness_scores[i] * (fitness_diff / score_diff)
            new_position = agents[i] - r1 * secant_step + r2 * (np.random.rand(n_clusters, n_features) - 0.5)
        else:
            new_position = agents[i] + r1 * (global_best - agents[i])

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
    global_best_score = fitness_scores[best_idx]

    print(f"Iteration {iteration+1}/{max_iter} - Best SSE: {global_best_score:.6f}")

# ==========================================================
# 🔹 6. FINAL K-MEANS USING SOA CENTROIDS
# ==========================================================

final_centroids, labels = kmeans_from_centroids(X_scaled, global_best)

clusters = np.unique(labels)
features = X.columns.tolist()

print("\n================ SOA + K-MEANS DONE ================")
print(pd.Series(labels).value_counts())
print("=====================================================\n")

# ==========================================================
# 🔹 7. KRUSKAL + MEDIAN & IQR
# ==========================================================

results = []

for i in range(len(features)):

    cluster_data = []
    medians = []
    iqrs = []

    for c in clusters:
        cluster_values = X_scaled[labels == c, i]
        cluster_data.append(cluster_values)

        median = np.median(cluster_values)
        q1 = np.percentile(cluster_values, 25)
        q3 = np.percentile(cluster_values, 75)
        iqr = q3 - q1

        medians.append(median)
        iqrs.append(iqr)

    stat, p_value = kruskal(*cluster_data)

    if p_value < 0.05:
        results.append((features[i], medians, iqrs))

# ==========================================================
# 🔹 8. GENERATE ONE SINGLE IMAGE
# ==========================================================

if len(results) == 0:
    print("No significant features found.")
else:

    rows = len(results)
    plt.figure(figsize=(8, 4 * rows))

    for idx, (feature_name, medians, iqrs) in enumerate(results):
        plt.subplot(rows, 1, idx + 1)

        x = np.arange(len(clusters))
        plt.bar(x, medians)
        plt.errorbar(x, medians, yerr=iqrs, fmt='none')

        plt.xticks(x, [f"C{c}" for c in clusters])
        plt.title(f"Median ± IQR (SOA + K-Means) - {feature_name} # Dataset") # Change
        plt.ylabel("Median Value")

    plt.tight_layout()

    output_file = "SOA+KMeans.jpg" # Change
    plt.savefig(output_file, format="jpg")
    plt.close()

    print(f"Single image saved as: {output_file}")