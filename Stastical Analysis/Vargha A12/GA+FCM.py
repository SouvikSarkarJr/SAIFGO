# ==========================================================
# GA + FUZZY C-MEANS (FCM) + KRUSKAL-WALLIS + VARGHA-DELANY A12
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import random
from sklearn.preprocessing import StandardScaler
from scipy.stats import kruskal
from scipy.spatial.distance import cdist
import os

# ==========================================================
# 🔹 1. DATASET PATH
# ==========================================================
file_path = "#"  # Change
n_clusters = 3
m = 2.0

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
# 🔹 3. HELPER FUNCTIONS
# ==========================================================

def compute_membership(X, centroids, m):
    dist = cdist(X, centroids)
    dist = np.fmax(dist, np.finfo(np.float64).eps)

    power = 2 / (m - 1)
    temp = dist ** power
    denominator = temp[:, :, None] / temp[:, None, :]
    u = 1 / np.sum(denominator, axis=2)

    return u.T


def xie_beni_index(X, u, centroids, m):
    n = X.shape[0]
    dist = cdist(X, centroids) ** 2
    numerator = np.sum((u.T ** m) * dist)

    centroid_distances = cdist(centroids, centroids)
    np.fill_diagonal(centroid_distances, np.inf)
    min_centroid_dist = np.min(centroid_distances) ** 2

    return numerator / (n * min_centroid_dist)


def fitness(centroids):
    u0 = compute_membership(X_scaled, centroids, m)

    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        X_scaled.T,
        c=n_clusters,
        m=m,
        error=0.005,
        maxiter=300,
        init=u0,
        seed=42
    )

    return xie_beni_index(X_scaled, u, cntr, m)


# ==========================================================
# 🔹 4. GENETIC ALGORITHM
# ==========================================================

def initialize_population():
    population = []
    for _ in range(pop_size):
        centroids = np.random.uniform(
            low=X_scaled.min(axis=0),
            high=X_scaled.max(axis=0),
            size=(n_clusters, n_features)
        )
        population.append(centroids)
    return population


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
    print(f"Generation {gen+1}/{generations} - Best XB: {min(scores):.6f}")

# Best GA solution
final_scores = [fitness(ind) for ind in population]
best_centroids = population[np.argmin(final_scores)]

# ==========================================================
# 🔹 5. FINAL FCM USING GA CENTROIDS
# ==========================================================
u0 = compute_membership(X_scaled, best_centroids, m)

cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    X_scaled.T,
    c=n_clusters,
    m=m,
    error=0.005,
    maxiter=1000,
    init=u0,
    seed=42
)

labels = np.argmax(u, axis=0)
clusters = np.unique(labels)
features = X.columns.tolist()

print("\n============= GA + FCM CLUSTERING DONE =============")
print(pd.Series(labels).value_counts())
print("====================================================\n")

# ==========================================================
# 🔹 6. VARGHA-DELANY A12 FUNCTION (Correct Implementation)
# ==========================================================
def vargha_delaney_a12(x, y):
    m = len(x)
    n = len(y)
    ranks = np.argsort(np.argsort(np.concatenate([x, y]))) + 1
    r1 = np.sum(ranks[:m])
    A12 = (r1 - m * (m + 1) / 2) / (m * n)
    return A12


# ==========================================================
# 🔹 7. KRUSKAL + EFFECT SIZE MATRICES
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
# 🔹 8. GENERATE ONE SINGLE IMAGE
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
        plt.title(f"Vargha-Delaney A12 (GA+FCM) # Dataset - {feature_name}") # Change
        plt.xticks(range(len(clusters)), [f"C{c}" for c in clusters])
        plt.yticks(range(len(clusters)), [f"C{c}" for c in clusters])

    plt.tight_layout()

    output_file = "GA+FCM.jpg"  # Change
    plt.savefig(output_file, format="jpg")
    plt.close()

    print(f"Single image saved as: {output_file}")
