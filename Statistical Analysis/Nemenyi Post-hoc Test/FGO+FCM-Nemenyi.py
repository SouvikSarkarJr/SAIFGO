import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikit_posthocs as sp
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from scipy.stats import kruskal
from scipy.spatial.distance import cdist
import os

# ==========================================================
# 🔹 1. DATASET PATH
# ==========================================================
file_path = "/Users/souvik/Documents/Project-II/Data-sets/Iris.csv" # Change
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

        # Mycelium growth toward best
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
clusters = np.unique(labels)
features = X.columns.tolist()

print("\n============= FGO + FCM CLUSTERING DONE =============")
print(pd.Series(labels).value_counts())
print("=====================================================\n")

# ==========================================================
# 🔹 6. KRUSKAL + NEMENYI
# ==========================================================

significant_results = []

for i in range(len(features)):

    cluster_data = []
    for c in clusters:
        cluster_values = X_scaled[labels == c, i]
        cluster_data.append(cluster_values)

    stat, p_value = kruskal(*cluster_data)

    if p_value < 0.05:

        temp_df = pd.DataFrame({
            "values": np.concatenate(cluster_data),
            "groups": np.concatenate([
                np.repeat(f"C{c}", len(cluster_data[idx]))
                for idx, c in enumerate(clusters)
            ])
        })

        nemenyi = sp.posthoc_nemenyi(
            temp_df,
            val_col="values",
            group_col="groups"
        )

        significant_results.append((features[i], nemenyi))

# ==========================================================
# 🔹 7. GENERATE ONE COMBINED IMAGE
# ==========================================================

if len(significant_results) == 0:
    print("No significant features found.")
else:

    rows = len(significant_results)
    plt.figure(figsize=(6, 4 * rows))

    for idx, (feature_name, nemenyi_matrix) in enumerate(significant_results):
        plt.subplot(rows, 1, idx + 1)
        plt.imshow(nemenyi_matrix)
        plt.colorbar()
        plt.title(f"Nemenyi Post-Hoc (FGO + FCM) Iris Dataset - {feature_name}") # Change
        plt.xticks(range(len(clusters)), [f"C{c}" for c in clusters])
        plt.yticks(range(len(clusters)), [f"C{c}" for c in clusters])

    plt.tight_layout()

    output_file = "Iris_FGO_FCM_Nemenyi.jpg" # Change
    plt.savefig(output_file, format="jpg")
    plt.close()

    print(f"Single image saved as: {output_file}")