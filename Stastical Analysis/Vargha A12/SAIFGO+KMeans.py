# ======================================================
#   FAST SAIFGO + KMeans
#   + KRUSKAL-WALLIS + VARGHA-DELANY A12
# ======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import levy, kruskal
import random
import os


# ======================================================
# Fast Vectorized Distance
# ======================================================

def fast_distance(X, centers):
    return np.sqrt(
        np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    )


# ======================================================
# Chaotic Initialization
# ======================================================

def chaotic_initialization(pop, dim, lb, ub):
    X = np.zeros((pop, dim))
    r = np.random.rand()

    for i in range(pop):
        r = 4 * r * (1 - r)
        X[i] = lb + r * (ub - lb)

    return X


def levy_flight(dim):
    return levy.rvs(size=dim)


# ======================================================
# SAIFGO + KMeans Hybrid
# ======================================================

class SAIFGO_Clustering:

    def __init__(self, n_clusters=3, pop_size=25, max_iter=60):
        self.n_clusters = n_clusters
        self.pop_size = pop_size
        self.max_iter = max_iter

    def fitness(self, X, centers):

        dist_matrix = fast_distance(X, centers)
        labels = np.argmin(dist_matrix, axis=1)

        if len(np.unique(labels)) < self.n_clusters:
            return -1e9

        return -np.sum(np.min(dist_matrix, axis=1))


    def fit(self, X):

        n_samples, n_features = X.shape
        dim = self.n_clusters * n_features
        lb, ub = X.min(), X.max()

        population = chaotic_initialization(
            self.pop_size, dim, lb, ub
        )

        best_fitness = -np.inf
        best_solution = None

        for iteration in range(self.max_iter):

            r = 1 - (iteration / self.max_iter) ** 2 * 0.9
            fitness_values = np.zeros(self.pop_size)

            for i in range(self.pop_size):

                centers = population[i].reshape(
                    self.n_clusters, n_features
                )

                fit = self.fitness(X, centers)
                fitness_values[i] = fit

                if fit > best_fitness:
                    best_fitness = fit
                    best_solution = population[i].copy()

            best_idx = np.argmax(fitness_values)
            worst_idx = np.argsort(fitness_values)[
                :int(0.2 * self.pop_size)
            ]

            for i in range(self.pop_size):

                if i == best_idx:
                    continue

                population[i] += r * (
                    best_solution - population[i]
                )

                if random.random() < 0.2:
                    population[i] += 0.01 * levy_flight(dim)

                population[i] = np.clip(population[i], lb, ub)

            for idx in worst_idx:
                population[idx] = lb + ub - population[idx]

        print("SAIFGO Optimization Completed.")

        initial_centers = best_solution.reshape(
            self.n_clusters, n_features
        )

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            init=initial_centers,
            n_init=1,
            max_iter=300,
            random_state=42
        )

        kmeans.fit(X)

        self.labels_ = kmeans.labels_
        return self


# ======================================================
# VARGHA-DELANY A12
# ======================================================

def vargha_delaney_a12(x, y):
    m = len(x)
    n = len(y)
    combined = np.concatenate([x, y])
    ranks = combined.argsort().argsort() + 1
    r1 = np.sum(ranks[:m])
    A12 = (r1 / m - (m + 1) / 2) / n
    return A12


# ======================================================
# MAIN EXECUTION
# ======================================================

if __name__ == "__main__":

    file_path = "#"   # Change
    n_clusters = 3

    if not os.path.exists(file_path):
        raise FileNotFoundError("Dataset not found.")

    df = pd.read_csv(file_path)

    X = df.select_dtypes(include=[np.number])
    X = X.drop(columns=["Id", "ID", "id"], errors="ignore")

    features = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SAIFGO_Clustering(
        n_clusters=n_clusters,
        pop_size=25,
        max_iter=60
    )

    model.fit(X_scaled)
    labels = model.labels_

    clusters = np.unique(labels)

    # ======================================================
    # KRUSKAL + VARGHA-DELANY A12
    # ======================================================

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


    # ======================================================
    # SINGLE IMAGE OUTPUT
    # ======================================================

    if len(effect_matrices) == 0:
        print("No significant features found.")
    else:

        rows = len(effect_matrices)
        plt.figure(figsize=(6, 4 * rows))

        for idx, (feature_name, matrix) in enumerate(effect_matrices):
            plt.subplot(rows, 1, idx + 1)
            plt.imshow(matrix)
            plt.colorbar()
            plt.title(f"Vargha-Delaney A12 Effect Size (# Dataset) - {feature_name} (SAIFGO+KMeans)") # Change
            plt.xticks(range(len(clusters)), [f"C{c}" for c in clusters])
            plt.yticks(range(len(clusters)), [f"C{c}" for c in clusters])

        plt.tight_layout()

        output_file = "SAIFGO+KMeans.jpg" # Change
        plt.savefig(output_file, format="jpg")
        plt.close()

        print(f"Single image saved as: {output_file}")