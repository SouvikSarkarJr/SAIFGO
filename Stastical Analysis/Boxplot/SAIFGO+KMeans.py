# ==========================================================
# SAIFGO + K-MEANS + BOXPLOT VISUALIZATION
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import levy
import os
import random


# ==========================================================
# 1️⃣ DATASET PATH
# ==========================================================
file_path = "#"   # Change
n_clusters = 3
pop_size = 30
max_iter = 50


# ==========================================================
# 2️⃣ LOAD DATA
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
# 3️⃣ DISTANCE FUNCTION
# ==========================================================
def fast_distance(X, centers):
    return np.sqrt(
        np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    )

def levy_flight(dim):
    return levy.rvs(size=dim)


# ==========================================================
# 4️⃣ SAIFGO OPTIMIZATION
# ==========================================================

dim = n_clusters * n_features
lb, ub = X_scaled.min(), X_scaled.max()

population = np.random.uniform(lb, ub, (pop_size, dim))

best_solution = None
best_sse = np.inf

for iteration in range(max_iter):

    for i in range(pop_size):

        centers = population[i].reshape(n_clusters, n_features)
        distances = fast_distance(X_scaled, centers)
        labels = np.argmin(distances, axis=1)

        sse = 0
        for k in range(n_clusters):
            cluster_points = X_scaled[labels == k]
            if len(cluster_points) > 0:
                sse += np.sum((cluster_points - centers[k])**2)

        if sse < best_sse:
            best_sse = sse
            best_solution = population[i].copy()

    r = 1 - (iteration / max_iter) ** 2 * 0.9

    for i in range(pop_size):

        population[i] += r * (best_solution - population[i])
        population[i] += 0.03 * levy_flight(dim)
        population[i] = np.clip(population[i], lb, ub)

print("SAIFGO Optimization Completed.")


# ==========================================================
# 5️⃣ K-MEANS REFINEMENT
# ==========================================================

initial_centers = best_solution.reshape(n_clusters, n_features)

kmeans = KMeans(
    n_clusters=n_clusters,
    init=initial_centers,
    n_init=1,
    max_iter=300
)

labels = kmeans.fit_predict(X_scaled)

print("K-Means Refinement Completed.")


# ==========================================================
# 6️⃣ BOXPLOT VISUALIZATION (SAME STYLE AS YOUR CODE)
# ==========================================================

X_original = pd.DataFrame(X, columns=X.columns)
X_original["Cluster"] = labels

plt.figure(figsize=(8, 6))

# Use first feature for visualization (same logic as yours)
feature_name = X.columns[0]

data_to_plot = [
    X_original[X_original["Cluster"] == c][feature_name]
    for c in range(n_clusters)
]

plt.boxplot(data_to_plot)
plt.xlabel("Clusters")
plt.ylabel(feature_name)
plt.title(f"Boxplot of {feature_name} Across Clusters (# Dataset)") # Change
plt.xticks(range(1, n_clusters + 1), [f"C{c}" for c in range(n_clusters)])
plt.tight_layout()

output_file = "SAIFGO+KMeans.jpg" # Change
plt.savefig(output_file, format="jpg")
plt.close()

print(f"Boxplot saved as: {output_file}")