# ======================================================
#   IFGO + KMeans Hybrid
#   + SSE Convergence Curve (K-Means Style)
#   (ONE IMAGE OUTPUT)
# ======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import levy
import random
import os


# ======================================================
# 1️⃣ DATASET PATH
# ======================================================
file_path = "Data-sets/Bank.csv"   # Change if needed
n_clusters = 3
max_iter = 50
pop_size = 30


# ======================================================
# 2️⃣ LOAD DATA
# ======================================================
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


# ======================================================
# 3️⃣ DISTANCE FUNCTION
# ======================================================
def fast_distance(X, centers):
    return np.sqrt(
        np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    )


def levy_flight(dim):
    return levy.rvs(size=dim)


# ======================================================
# 4️⃣ SAIFGO OPTIMIZATION (TRACK SSE)
# ======================================================

dim = n_clusters * n_features
lb, ub = X.min(), X.max()

population = np.random.uniform(lb, ub, (pop_size, dim))

best_solution = None
best_sse = np.inf
sse_history = []

for iteration in range(max_iter):

    sse_values = np.zeros(pop_size)

    for i in range(pop_size):

        centers = population[i].reshape(n_clusters, n_features)
        distances = fast_distance(X, centers)
        labels = np.argmin(distances, axis=1)

        # Compute SSE
        sse = 0
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                sse += np.sum((cluster_points - centers[k])**2)

        sse_values[i] = sse

        if sse < best_sse:
            best_sse = sse
            best_solution = population[i].copy()

    sse_history.append(best_sse)

    best_idx = np.argmin(sse_values)
    r = 1 - (iteration / max_iter) ** 2 * 0.9

    for i in range(pop_size):

        if i == best_idx:
            continue

        # Attraction toward best
        population[i] += r * (best_solution - population[i])

        # Lévy exploration
        population[i] += 0.03 * levy_flight(dim)

        population[i] = np.clip(population[i], lb, ub)


print("SAIFGO Optimization Completed.")


# ======================================================
# 5️⃣ KMeans REFINEMENT
# ======================================================

initial_centers = best_solution.reshape(n_clusters, n_features)

kmeans = KMeans(
    n_clusters=n_clusters,
    init=initial_centers,
    n_init=1,
    max_iter=300
)

kmeans.fit(X)

final_sse = kmeans.inertia_
sse_history.append(final_sse)

print("KMeans Refinement Completed.")


# ======================================================
# 6️⃣ PLOT CONVERGENCE CURVE
# ======================================================

plt.figure()
plt.plot(range(1, len(sse_history)+1), sse_history)
plt.xlabel("Iteration")
plt.ylabel("Inertia (SSE)")
plt.title("SAIFGO + KMeans Convergence Curve (Bank Dataset)") # Change
plt.tight_layout()

output_file = "SAIFGO+KMeans.jpg" # Change
plt.savefig(output_file, format="jpg")
plt.close()

print(f"Convergence curve saved as: {output_file}")