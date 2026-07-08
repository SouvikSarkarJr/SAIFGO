# ======================================================
#   FAST SAIFGO + KMeans
#   + KRUSKAL-WALLIS + NEMENYI
# ======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikit_posthocs as sp
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

        self.best_centers = kmeans.cluster_centers_
        self.labels_ = kmeans.labels_

        print("K-Means Refinement Completed.")
        return self


# ======================================================
# MAIN EXECUTION
# ======================================================

if __name__ == "__main__":

    file_path = "#"   # Change if needed
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
    # KRUSKAL + NEMENYI
    # ======================================================

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


    # ======================================================
    # SINGLE IMAGE OUTPUT
    # ======================================================

    if len(significant_results) == 0:
        print("No significant features found.")
    else:

        rows = len(significant_results)
        plt.figure(figsize=(6, 4 * rows))

        for idx, (feature_name, nemenyi_matrix) in enumerate(significant_results):
            plt.subplot(rows, 1, idx + 1)
            plt.imshow(nemenyi_matrix)
            plt.colorbar() 
            plt.title(f"Nemenyi Post-Hoc (# Dataset) - {feature_name} (SAIFGO)") # Change
            plt.xticks(range(len(clusters)), [f"C{c}" for c in clusters])
            plt.yticks(range(len(clusters)), [f"C{c}" for c in clusters])

        plt.tight_layout()

        output_file = "SAIFGO+KMeans.jpg" # Change
        plt.savefig(output_file, format="jpg")
        plt.close()

        print(f"Single image saved as: {output_file}")