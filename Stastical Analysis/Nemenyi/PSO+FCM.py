# ==========================================================
# PSO + FUZZY C-MEANS (FCM) + KRUSKAL-WALLIS + NEMENYI
# ==========================================================

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
file_path = "#"  #Change
n_clusters = 3
m = 2.0

# PSO parameters
n_particles = 20
max_iter = 30
w = 0.7
c1 = 1.5
c2 = 1.5

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
# 🔹 4. PARTICLE SWARM OPTIMIZATION
# ==========================================================

particles = np.random.uniform(
    low=X_scaled.min(axis=0),
    high=X_scaled.max(axis=0),
    size=(n_particles, n_clusters, n_features)
)

velocities = np.zeros_like(particles)

personal_best = particles.copy()
personal_best_scores = np.array([fitness(p) for p in particles])

global_best_index = np.argmin(personal_best_scores)
global_best = personal_best[global_best_index].copy()

for iteration in range(max_iter):

    for i in range(n_particles):

        r1 = np.random.rand()
        r2 = np.random.rand()

        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (personal_best[i] - particles[i])
            + c2 * r2 * (global_best - particles[i])
        )

        particles[i] = particles[i] + velocities[i]

        score = fitness(particles[i])

        if score < personal_best_scores[i]:
            personal_best[i] = particles[i].copy()
            personal_best_scores[i] = score

    global_best_index = np.argmin(personal_best_scores)
    global_best = personal_best[global_best_index].copy()

    print(f"Iteration {iteration+1}/{max_iter} - Best XB: {personal_best_scores[global_best_index]:.6f}")

# ==========================================================
# 🔹 5. FINAL FCM USING PSO CENTROIDS
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

print("\n============= PSO + FCM CLUSTERING DONE =============")
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
        plt.title(f"Nemenyi Post-Hoc (PSO+FCM) # Dataset - {feature_name}") #Change
        plt.xticks(range(len(clusters)), [f"C{c}" for c in clusters])
        plt.yticks(range(len(clusters)), [f"C{c}" for c in clusters])

    plt.tight_layout()

    output_file = "PSO+FCM.jpg"  #Change
    plt.savefig(output_file, format="jpg")
    plt.close()

    print(f"Single image saved as: {output_file}")
