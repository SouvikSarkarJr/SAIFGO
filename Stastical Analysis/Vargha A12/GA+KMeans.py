# ==========================================================
# GA + K-MEANS + KRUSKAL-WALLIS + VARGHA-DELANY A12
# (ONE SINGLE IMAGE OUTPUT)
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from scipy.stats import kruskal
from scipy.spatial.distance import cdist
import os

# ==========================================================
# 🔹 1. DATASET PATH
# ==========================================================
file_path = "#"  # change
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
# 🔹 5. GENETIC ALGORITHM
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

best_centroids = population[np.argmin([fitness(ind) for ind in population])]

# ==========================================================
# 🔹 6. FINAL K-MEANS USING GA CENTROIDS
# ==========================================================

final_centroids, labels = kmeans_from_centroids(X_scaled, best_centroids)

clusters = np.unique(labels)
features = X.columns.tolist()

print("\n================ GA + K-MEANS DONE ================")
print(pd.Series(labels).value_counts())
print("===================================================\n")

# ==========================================================
# 🔹 7. DEFINE VARGHA-DELANY A12 (Correct Formula)
# ==========================================================

def vargha_delaney_a12(x, y):
    m = len(x)
    n = len(y)
    ranks = np.argsort(np.argsort(np.concatenate([x, y]))) + 1
    r1 = np.sum(ranks[:m])
    A12 = (r1 - m * (m + 1) / 2) / (m * n)
    return A12

# ==========================================================
# 🔹 8. KRUSKAL + EFFECT SIZE MATRICES
# ==========================================================

effect_matrices = []

for i in range(len(features)):

    cluster_data = []
    for c in clusters:
        cluster_values = X_scaled[labels == c, i]
        cluster_data.append(cluster_values)

    stat, p_value = kruskal(*cluster_data)

    if p_value < 0.05:

        size = len(clusters)
        matrix = np.zeros((size, size))

        for a in range(size):
            for b in range(size):
                if a != b:
                    matrix[a, b] = vargha_delaney_a12(
                        cluster_data[a],
                        cluster_data[b]
                    )

        effect_matrices.append((features[i], matrix))

# ==========================================================
# 🔹 9. GENERATE ONE SINGLE IMAGE
# ==========================================================

if len(effect_matrices) == 0:
    print("No significant features found.")
else:

    rows = len(effect_matrices)
    plt.figure(figsize=(6, 4 * rows))

    for idx, (feature_name, matrix) in enumerate(effect_matrices):
        plt.subplot(rows, 1, idx + 1)
        plt.imshow(matrix)
        plt.colorbar()
        plt.title(f"Vargha-Delaney A12 (GA + K-Means) # Dataset- {feature_name}")
        plt.xticks(range(len(clusters)), [f"C{c}" for c in clusters])
        plt.yticks(range(len(clusters)), [f"C{c}" for c in clusters])

    plt.tight_layout()

    output_file = "GA+KMeans.jpg"  # change
    plt.savefig(output_file, format="jpg")
    plt.close()

    print(f"Single image saved as: {output_file}")
