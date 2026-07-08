# ==========================================================
# GT + FUZZY C-MEANS (FCM) + WILCOXON TEST + JPG GRAPH
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import random
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon
from scipy.spatial.distance import cdist
import os

# ==========================================================
# 🔹 1. DATASET SETTINGS
# ==========================================================
file_path = "#"  # change
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
# 🔹 4. GENERAL TREND (GT) ALGORITHM
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

# Best GT solution
final_scores = [fitness(ind) for ind in population]
best_centroids = population[np.argmin(final_scores)]

# ==========================================================
# 🔹 5. FINAL FCM USING GT CENTROIDS
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
df["Cluster"] = labels

print("\n============= GT + FCM CLUSTERING DONE =============")
print(df["Cluster"].value_counts())
print("====================================================\n")

# ==========================================================
# 🔹 6. WILCOXON SIGNED-RANK TEST
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
# 🔹 7. GENERATE GRAPH
# ==========================================================
plt.figure()
plt.bar(features, p_values)
plt.axhline(y=0.05)
plt.xticks(rotation=45)
plt.title("Wilcoxon Signed-Rank Test (GT + FCM)")
plt.xlabel("Features (# Dataset)") # change
plt.ylabel("P-values")
plt.tight_layout()

output_file = "GT+FCM.jpg"  # change
plt.savefig(output_file, format="jpg")
plt.close()

print(f"Graph saved successfully as: {output_file}")