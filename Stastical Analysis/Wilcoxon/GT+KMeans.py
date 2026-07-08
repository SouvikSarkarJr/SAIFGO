# ==========================================================
# GT + K-MEANS + WILCOXON SIGNED-RANK TEST + JPG GRAPH
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon
from scipy.spatial.distance import cdist
import os

# ==========================================================
# 🔹 1. SETTINGS
# ==========================================================
file_path = "#"  # CHANGE
n_clusters = 3

pop_size = 20
generations = 30
mutation_rate = 0.1

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
# 🔹 5. GENERAL TREND (GT) ALGORITHM
# ==========================================================

def initialize_population():
    return [
        np.random.uniform(
            low=X_scaled.min(axis=0),
            high=X_scaled.max(axis=0),
            size=(n_clusters, n_features)
        )
        for _ in range(pop_size)
    ]

def selection(population, scores):
    idx = np.argsort(scores)
    return [population[i] for i in idx[:pop_size // 2]]

def crossover(p1, p2):
    alpha = np.random.rand()
    return alpha * p1 + (1 - alpha) * p2

def mutation(centroids):
    if random.random() < mutation_rate:
        centroids += np.random.normal(0, 0.1, centroids.shape)
    return centroids

population = initialize_population()

for gen in range(generations):
    scores = [fitness(ind) for ind in population]
    selected = selection(population, scores)

    new_population = selected.copy()

    while len(new_population) < pop_size:
        p1, p2 = random.sample(selected, 2)
        child = crossover(p1, p2)
        child = mutation(child)
        new_population.append(child)

    population = new_population
    print(f"Generation {gen+1}/{generations} - Best SSE: {min(scores):.6f}")

# Best GT centroids
best_centroids = population[np.argmin([fitness(ind) for ind in population])]

# ==========================================================
# 🔹 6. FINAL K-MEANS USING GT CENTROIDS
# ==========================================================

final_centroids, labels = kmeans_from_centroids(X_scaled, best_centroids)
df["Cluster"] = labels

print("\n================ GT + K-MEANS DONE ================")
print(df["Cluster"].value_counts())
print("===================================================\n")

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
plt.title("Wilcoxon Signed-Rank Test (GT + K-Means)")
plt.xlabel("Features (# Dataset)") #Change
plt.ylabel("P-values")
plt.tight_layout()

output_file = "GT+KMeans.jpg"  # CHANGE
plt.savefig(output_file, format="jpg")
plt.close()

print(f"Graph saved successfully as: {output_file}")