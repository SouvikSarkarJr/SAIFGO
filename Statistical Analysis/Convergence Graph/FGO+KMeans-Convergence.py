import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import os

# ==========================================================
# 🔹 1. DATASET PATH
# ==========================================================
file_path = "/Users/souvik/Documents/Project-II/Data-sets/Placement.csv" # Change
n_clusters = 3
max_iter = 50

# FGO parameters
n_fungi = 20
fgo_iterations = 30
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
# 🔹 5. FUNGAL GROWTH OPTIMIZATION (FGO) PHASE
# ==========================================================

fungi = np.random.uniform(
    low=X.min(axis=0),
    high=X.max(axis=0),
    size=(n_fungi, n_clusters, n_features)
)

fitness_scores = np.array([fitness(f) for f in fungi])
best_idx = np.argmin(fitness_scores)
global_best = fungi[best_idx].copy()

fgo_history = []

for iteration in range(fgo_iterations):

    fgo_history.append(np.min(fitness_scores))

    for i in range(n_fungi):

        # Mycelium growth toward best fungus
        growth = growth_rate * np.random.rand() * (global_best - fungi[i])

        # Spore mutation
        spore = spore_rate * np.random.randn(n_clusters, n_features)

        new_position = fungi[i] + growth + spore

        new_position = np.clip(
            new_position,
            X.min(axis=0),
            X.max(axis=0)
        )

        new_score = fitness(new_position)

        if new_score < fitness_scores[i]:
            fungi[i] = new_position
            fitness_scores[i] = new_score

    best_idx = np.argmin(fitness_scores)
    global_best = fungi[best_idx].copy()

    print(f"Iteration {iteration+1}/{fgo_iterations} - Best SSE: {fitness_scores[best_idx]:.6f}")

# ==========================================================
# 🔹 6. FINAL K-MEANS REFINEMENT
# ==========================================================

final_centroids, labels, kmeans_history = kmeans_from_centroids(
    X, global_best, max_iter=max_iter
)

# ==========================================================
# 🔹 7. PLOT COMBINED CONVERGENCE CURVE
# ==========================================================

plt.figure()

# FGO phase
plt.plot(range(1, len(fgo_history) + 1),
         fgo_history,
         label="FGO Phase")

# K-Means refinement phase
kmeans_x = range(len(fgo_history) + 1,
                 len(fgo_history) + len(kmeans_history) + 1)

plt.plot(kmeans_x,
         kmeans_history,
         label="K-Means Refinement Phase")

plt.xlabel("Iteration")
plt.ylabel("Inertia (SSE)")
plt.title("FGO + K-Means Convergence Curve (Placement Dataset)") # Change
plt.legend()
plt.tight_layout()

output_file = "Placement_FGO_KMeans_Convergence.jpg" # Change
plt.savefig(output_file, format="jpg")
plt.close()

print(f"Convergence curve saved as: {output_file}")