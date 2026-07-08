import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from scipy.stats import friedmanchisquare
from scipy.spatial.distance import cdist
import os

# ==========================================================
# 🔹 1. DATASET SETTINGS
# ==========================================================
file_path = "#" # Change
n_clusters = 3
m = 2.0

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
# 🔹 4. FUNGAL GROWTH OPTIMIZATION (FGO)
# ==========================================================

fungi = np.random.uniform(
    low=X_scaled.min(axis=0),
    high=X_scaled.max(axis=0),
    size=(n_fungi, n_clusters, n_features)
)

fitness_scores = np.array([fitness(f) for f in fungi])

best_idx = np.argmin(fitness_scores)
best_solution = fungi[best_idx].copy()

for iteration in range(max_iter):

    for i in range(n_fungi):

        # Mycelium expansion toward best fungus
        growth = growth_rate * np.random.rand() * (best_solution - fungi[i])

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
    best_solution = fungi[best_idx].copy()

    print(f"Iteration {iteration+1}/{max_iter} - Best XB: {fitness_scores[best_idx]:.6f}")

# ==========================================================
# 🔹 5. FINAL FCM USING FGO CENTROIDS
# ==========================================================

u0 = compute_membership(X_scaled, best_solution, m)

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

print("\n============= FGO + FCM CLUSTERING DONE =============")
print(df["Cluster"].value_counts())
print("====================================================\n")

# ==========================================================
# 🔹 6. FRIEDMAN TEST
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
# 🔹 7. GENERATE GRAPH
# ==========================================================

plt.figure()
plt.bar(features, p_values)
plt.axhline(y=0.05)
plt.xticks(rotation=45)
plt.title("Friedman Test (FGO + FCM)")
plt.xlabel("Features (# Dataset)") # Change
plt.ylabel("P-values")
plt.tight_layout()

output_file = "FGO+FCM.jpg" # Change
plt.savefig(output_file, format="jpg")
plt.close()

print(f"Graph saved successfully as: {output_file}")