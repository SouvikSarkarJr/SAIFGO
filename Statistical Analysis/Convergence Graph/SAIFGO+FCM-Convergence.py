# ==========================================================
# IFGO + FCM + CONVERGENCE CURVE ANALYSIS (ONE IMAGE)
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import levy
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
# 2️⃣ CHAOTIC INITIALIZATION
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
# 3️⃣ SAIFGO OPTIMIZATION
# ==========================================================

def amfgo_optimize(X, n_clusters=3, pop_size=25, max_iter=60):

    n_samples, n_features = X.shape
    dim = n_clusters * n_features
    lb, ub = X.min(), X.max()

    population = chaotic_initialization(pop_size, dim, lb, ub)

    best_fitness = -np.inf
    best_solution = None

    for iteration in range(max_iter):

        r = 1 - (iteration / max_iter) ** 2 * 0.9
        fitness_values = np.zeros(pop_size)

        for i in range(pop_size):

            centers = population[i].reshape(n_clusters, n_features)
            dist_matrix = fast_distance(X, centers)
            fitness = -np.sum(np.min(dist_matrix, axis=1))

            fitness_values[i] = fitness

            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = population[i].copy()

        best_idx = np.argmax(fitness_values)

        for i in range(pop_size):

            if i == best_idx:
                continue

            population[i] += r * (best_solution - population[i])

            if random.random() < 0.2:
                population[i] += 0.01 * levy_flight(dim)

            population[i] = np.clip(population[i], lb, ub)

    print("SAIFGO Optimization Completed.")

    return best_solution.reshape(n_clusters, n_features)


# ==========================================================
# 4️⃣ FCM WITH CONVERGENCE TRACKING
# ==========================================================

def fcm_with_convergence(X, centers, m=2, max_iter=50, epsilon=1e-5):

    n_samples = X.shape[0]
    n_clusters = centers.shape[0]

    # Initialize membership
    U = np.random.dirichlet(np.ones(n_clusters), size=n_samples)

    objective_history = []

    for iteration in range(max_iter):

        U_m = U ** m
        centers = (U_m.T @ X) / np.sum(U_m.T, axis=1, keepdims=True)

        dist = fast_distance(X, centers) + 1e-10

        # Objective function Jm
        J = np.sum((U ** m) * (dist ** 2))
        objective_history.append(J)

        new_U = np.zeros_like(U)

        for i in range(n_samples):
            for k in range(n_clusters):
                denominator = np.sum(
                    (dist[i, k] / dist[i, :]) ** (2 / (m - 1))
                )
                new_U[i, k] = 1 / denominator

        if np.linalg.norm(new_U - U) < epsilon:
            print(f"FCM Converged at iteration {iteration}")
            break

        U = new_U

    return objective_history


# ==========================================================
# 5️⃣ MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":

    file_path = "Data-sets/Bank.csv"  # Change

    if not os.path.exists(file_path):
        raise FileNotFoundError("Dataset not found.")

    df = pd.read_csv(file_path)

    X = df.select_dtypes(include=[np.number])
    X = X.drop(columns=["Id", "ID", "id"], errors="ignore")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 1: AMFGO
    initial_centers = amfgo_optimize(X_scaled)

    # Step 2: FCM convergence tracking
    objective_history = fcm_with_convergence(
        X_scaled,
        initial_centers,
        m=2,
        max_iter=50
    )

    # ======================================================
    # 6️⃣ PLOT CONVERGENCE CURVE
    # ======================================================

    plt.figure()
    plt.plot(range(1, len(objective_history) + 1), objective_history)
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function (Jm)")
    plt.title("SAIFGO + FCM Convergence Curve (Bank Dataset)")
    plt.tight_layout()

    save_path = os.path.join(os.getcwd(), "SAIFGO+FCM.jpg")
    plt.savefig(save_path, format="jpg", dpi=300)
    plt.close()

    print("\nConvergence curve saved at:")
    print(save_path)