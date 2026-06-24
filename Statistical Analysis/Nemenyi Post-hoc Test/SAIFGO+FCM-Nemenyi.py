# ==========================================================
# IFGO + FCM + KRUSKAL-WALLIS + NEMENYI
# (ONE SINGLE IMAGE OUTPUT)
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from sklearn.preprocessing import StandardScaler
from scipy.stats import levy, kruskal
import random
import os


# ==========================================================
# 1️⃣ FAST DISTANCE
# ==========================================================

def fast_distance(X, centers):
    return np.sqrt(
        np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    )


# ==========================================================
# 2️⃣ FCM REFINEMENT
# ==========================================================

def fcm_refinement(X, centers, m=2, max_iter=50):

    for _ in range(max_iter):

        dist = fast_distance(X, centers) + 1e-10
        inv_dist = 1 / dist

        U = inv_dist ** (2 / (m - 1))
        U = U / np.sum(U, axis=1, keepdims=True)

        new_centers = []

        for j in range(len(centers)):
            numerator = np.sum(
                (U[:, j] ** m).reshape(-1, 1) * X,
                axis=0
            )
            denominator = np.sum(U[:, j] ** m)
            new_centers.append(numerator / denominator)

        new_centers = np.array(new_centers)

        if np.linalg.norm(new_centers - centers) < 1e-5:
            break

        centers = new_centers

    labels = np.argmax(U, axis=1)
    return centers, labels


# ==========================================================
# 3️⃣ CHAOTIC INITIALIZATION
# ==========================================================

def chaotic_initialization(pop, dim, lb, ub):
    X = np.zeros((pop, dim))
    r = np.random.rand()

    for i in range(pop):
        r = 4 * r * (1 - r)
        X[i] = lb + r * (ub - lb)

    return X


def levy_flight(dim):
    return levy.rvs(size=dim)


# ==========================================================
# 4️⃣ SAIFGO + FCM CLASS
# ==========================================================

class AMFGO_Clustering:

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

            for i in range(self.pop_size):

                if i == best_idx:
                    continue

                population[i] += r * (
                    best_solution - population[i]
                )

                if random.random() < 0.2:
                    population[i] += 0.01 * levy_flight(dim)

                population[i] = np.clip(population[i], lb, ub)

        print("SAIFGO Optimization Completed.")

        initial_centers = best_solution.reshape(
            self.n_clusters, n_features
        )

        centers, labels = fcm_refinement(X, initial_centers)

        self.best_centers = centers
        self.labels_ = labels

        print("FCM Refinement Completed.")
        return self


# ==========================================================
# 5️⃣ MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":

    file_path = "Data-sets/Wave.csv"   # Change

    if not os.path.exists(file_path):
        raise FileNotFoundError("Dataset not found.")

    df = pd.read_csv(file_path)

    X = df.select_dtypes(include=[np.number])
    X = X.drop(columns=["Id", "ID", "id"], errors="ignore")

    features = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = AMFGO_Clustering(
        n_clusters=3,
        pop_size=25,
        max_iter=60
    )

    model.fit(X_scaled)

    labels = model.labels_
    clusters = np.unique(labels)

    significant_results = []

    # ======================================================
    # KRUSKAL + NEMENYI
    # ======================================================

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
    # SAVE SINGLE IMAGE
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
            plt.title(f"Nemenyi Post-Hoc (Wave Dataset) - {feature_name}") # Change
            plt.xticks(range(len(clusters)), [f"C{c}" for c in clusters])
            plt.yticks(range(len(clusters)), [f"C{c}" for c in clusters])

        plt.tight_layout()

        save_path = os.path.join(os.getcwd(), "SAIFGO+FCM+Wave.jpg") # Change
        plt.savefig(save_path, format="jpg", dpi=300)
        plt.close()

        print("\nImage successfully saved at:")
        print(save_path)