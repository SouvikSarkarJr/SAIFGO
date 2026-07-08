# ==========================================================
# GA + K-MEANS + CONVERGENCE CURVE ANALYSIS
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import os

# ==========================================================
# 🔹 1. DATASET PATH
# ==========================================================
file_path = "#"  #Change
n_clusters = 3
max_iter = 50

# GA parameters
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
X = scaler.fit_transform(X)

n_samples, n_features = X.shape

# ==========================================================
# 🔹 3. K-MEANS FUNCTION
# ==========================================================

def kmeans_from_centroids(X, centroids, max_iter=100):
    inertia_history = []

    for _ in range(max_iter):
        distances = cdist(X, centroids)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[labels == k].mean(axis=0) if np.any(labels == k)
            else centroids[k]
            for k in range(n_clusters)
        ])

        inertia = np.sum((distances[np.arange(n_samples), labels]) ** 2)
        inertia_history.append(inertia)

        if np.allclose(new_centroids, centroids):
            break

        centroids = new_centroids

    return centroids, labels, inertia_history

# ==========================================================
# 🔹 4. FITNESS FUNCTION (SSE)
# ==========================================================

def fitness(centroids):
    _, _, inertia_hist = kmeans_from_centroids(X, centroids, max_iter=20)
    return inertia_hist[-1]

# ==========================================================
# 🔹 5. GENETIC ALGORITHM PHASE
# ==========================================================

def initialize_population():
    return [
        np.random.uniform(
            low=X.min(axis=0),
            high=X.max(axis=0),
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
ga_history = []

for gen in range(generations):
    scores = [fitness(ind) for ind in population]
    ga_history.append(min(scores))

    selected = selection(population, scores)
    new_population = selected.copy()

    while len(new_population) < pop_size:
        p1, p2 = random.sample(selected, 2)
        child = crossover(p1, p2)
        child = mutation(child)
        new_population.append(child)

    population = new_population

best_centroids = population[np.argmin([fitness(ind) for ind in population])]

# ==========================================================
# 🔹 6. FINAL K-MEANS REFINEMENT
# ==========================================================

final_centroids, labels, kmeans_history = kmeans_from_centroids(
    X, best_centroids, max_iter=max_iter
)

# ==========================================================
# 🔹 7. PLOT COMBINED CONVERGENCE CURVE
# ==========================================================

plt.figure()

# GA phase
plt.plot(range(1, len(ga_history) + 1),
         ga_history,
         label="GA Phase")

# K-Means refinement phase (shift x-axis)
kmeans_x = range(len(ga_history) + 1,
                 len(ga_history) + len(kmeans_history) + 1)

plt.plot(kmeans_x,
         kmeans_history,
         label="K-Means Refinement Phase")

plt.xlabel("Iteration")
plt.ylabel("Inertia (SSE)")
plt.title("GA + K-Means Convergence Curve (# Dataset)") #Change
plt.legend()
plt.tight_layout()

output_file = "GA+KMeans.jpg"  #Change
plt.savefig(output_file, format="jpg")
plt.close()

print(f"Convergence curve saved as: {output_file}")
