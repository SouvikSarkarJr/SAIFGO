# ==========================================================
# GTO (Gorilla Troop Optimization) + FUZZY C-MEANS (FCM)
# + INDEPENDENT T-TEST + LINE GRAPH (JPG)
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
from scipy.spatial.distance import cdist
import os

# ==========================================================
# 🔹 1. DATASET SETTINGS
# ==========================================================
file_path = "#"  # change if needed
n_clusters = 3
m = 2.0

# GTO parameters
n_agents = 20
max_iter = 30

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
# 🔹 4. GORILLA TROOP OPTIMIZATION (GTO)
# ==========================================================

agents = np.random.uniform(
    low=X_scaled.min(axis=0),
    high=X_scaled.max(axis=0),
    size=(n_agents, n_clusters, n_features)
)

fitness_scores = np.array([fitness(a) for a in agents])
best_idx = np.argmin(fitness_scores)
silverback = agents[best_idx].copy()

for iteration in range(max_iter):

    for i in range(n_agents):

        r = np.random.rand()
        C = 2 * r - 1
        F = np.random.rand()

        if F < 0.5:
            new_position = agents[i] + C * (agents[i] - silverback)
        else:
            new_position = silverback - C * np.abs(silverback - agents[i])

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
    silverback = agents[best_idx].copy()

    print(f"Iteration {iteration+1}/{max_iter} - Best XB: {fitness_scores[best_idx]:.6f}")

# ==========================================================
# 🔹 5. FINAL FCM USING GTO CENTROIDS
# ==========================================================

u0 = compute_membership(X_scaled, silverback, m)

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

print("\n============= GTO + FCM CLUSTERING DONE =============")
print(df["Cluster"].value_counts())
print("=====================================================\n")

# ==========================================================
# 🔹 6. INDEPENDENT T-TEST (Welch’s)
# ==========================================================

clusters = np.unique(labels)

if len(clusters) < 2:
    raise ValueError("Need at least 2 clusters for T-Test.")

cluster_0 = X_scaled[labels == clusters[0]]
cluster_1 = X_scaled[labels == clusters[1]]

features = X.columns.tolist()
p_values = []

print("============= T-TEST RESULTS =============")

for i in range(len(features)):
    stat, p_value = ttest_ind(
        cluster_0[:, i],
        cluster_1[:, i],
        equal_var=False
    )

    p_values.append(p_value)

    print(f"\nFeature: {features[i]}")
    print(f"T-Statistic = {stat:.4f}")
    print(f"P-value     = {p_value:.6f}")

    if p_value < 0.05:
        print("Significant Difference")
    else:
        print("Not Significant")

print("==========================================\n")

# ==========================================================
# 🔹 7. GENERATE LINE GRAPH
# ==========================================================

plt.figure()

plt.plot(features, p_values, marker='o')
plt.axhline(y=0.05)

plt.xticks(rotation=45)
plt.title("T-Test (GTO + FCM)")
plt.xlabel("Features (# Dataset)")
plt.ylabel("P-values")

plt.tight_layout()

output_file = "GTO+FCM.jpg"
plt.savefig(output_file, format="jpg")
plt.close()

print(f"Graph saved successfully as: {output_file}")