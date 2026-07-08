# ==========================================================
# SOA (SECANT OPTIMIZATION ALGORITHM) + FCM
# + KRUSKAL-WALLIS + MEDIAN & IQR (ONE IMAGE)
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from scipy.stats import kruskal
from scipy.spatial.distance import cdist
import os

# ==========================================================
# 🔹 1. DATASET PATH
# ==========================================================
file_path = "#" # Change
n_clusters = 3
m = 2.0

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
# 🔹 4. SECANT OPTIMIZATION ALGORITHM (SOA)
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
        
        # Avoid division by zero if fitness values are identical
        fitness_diff = agents[i] - global_best
        score_diff = fitness_scores[i] - global_best_score
        
        r1 = np.random.rand()
        r2 = np.random.rand()

        if abs(score_diff) > 1e-6:
            # Secant-step update logic
            secant_step = fitness_scores[i] * (fitness_diff / score_diff)
            new_position = agents[i] - r1 * secant_step + r2 * (np.random.rand(n_clusters, n_features) - 0.5)
        else:
            # Fallback exploration behavior if slope cannot be determined
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

    print(f"Iteration {iteration+1}/{max_iter} - Best XB: {global_best_score:.6f}")

# ==========================================================
# 🔹 5. FINAL FCM USING SOA CENTROIDS
# ==========================================================

u0 = compute_membership(X_scaled, global_best, m)

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

print("\n============= SOA + FCM CLUSTERING DONE =============")
print(pd.Series(labels).value_counts())
print("=====================================================\n")

# ==========================================================
# 🔹 6. KRUSKAL + MEDIAN & IQR
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
# 🔹 7. GENERATE ONE SINGLE IMAGE
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
        plt.title(f"Median ± IQR (SOA + FCM) # Dataset - {feature_name}") # Change
        plt.ylabel("Median Value")

    plt.tight_layout()

    output_file = "SOA+FCM.jpg" # Change
    plt.savefig(output_file, format="jpg")
    plt.close()

    print(f"Single image saved as: {output_file}")