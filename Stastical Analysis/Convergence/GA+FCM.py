# ==========================================================
# GA + FCM + CONVERGENCE CURVE ANALYSIS
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import os

# ==========================================================
# 🔹 1. DATASET PATH
# ==========================================================
file_path = "#"  # Change
n_clusters = 3
m = 2.0
epsilon = 1e-5

# GA parameters
pop_size = 20
ga_generations = 30
mutation_rate = 0.1

# FCM parameters
fcm_max_iter = 50

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
# 🔹 3. HELPER FUNCTIONS
# ==========================================================

def compute_membership(X, centroids, m):
    dist = cdist(X, centroids) + 1e-10
    power = 2 / (m - 1)
    temp = dist ** power
    denominator = temp[:, :, None] / temp[:, None, :]
    u = 1 / np.sum(denominator, axis=2)
    return u

def objective_function(X, U, centroids, m):
    dist = cdist(X, centroids)
    return np.sum((U ** m) * (dist ** 2))

# ==========================================================
# 🔹 4. GENETIC ALGORITHM PHASE
# ==========================================================

def initialize_population():
    return [
        np.random.uniform(
            low=X.min(axis=0),
            high=X.max(axis=0),
            size=(n_clusters, n_features)
        )
        for _ in range(pop_size)
    ]

def fitness(centroids):
    U = compute_membership(X, centroids, m)
    return objective_function(X, U, centroids, m)

def selection(pop, scores):
    idx = np.argsort(scores)
    return [pop[i] for i in idx[:pop_size // 2]]

def crossover(p1, p2):
    alpha = np.random.rand()
    return alpha * p1 + (1 - alpha) * p2

def mutation(centroids):
    if random.random() < mutation_rate:
        centroids += np.random.normal(0, 0.1, centroids.shape)
    return centroids

population = initialize_population()
ga_history = []

for gen in range(ga_generations):
    scores = [fitness(ind) for ind in population]
    ga_history.append(min(scores))

    selected = selection(population, scores)
    new_population = selected.copy()

    while len(new_population) < pop_size:
        p1, p2 = random.sample(selected, 2)
        child = crossover(p1, p2)
        child = mutation(child)
        new_population.append(child)

    population = new_population

best_centroids = population[np.argmin([fitness(ind) for ind in population])]

# ==========================================================
# 🔹 5. FCM REFINEMENT PHASE (Manual)
# ==========================================================

U = compute_membership(X, best_centroids, m)
fcm_history = []

for iteration in range(fcm_max_iter):

    U_m = U ** m
    centroids = (U_m.T @ X) / np.sum(U_m.T, axis=1, keepdims=True)

    dist = cdist(X, centroids) + 1e-10
    J = np.sum((U ** m) * (dist ** 2))
    fcm_history.append(J)

    new_U = compute_membership(X, centroids, m)

    if np.linalg.norm(new_U - U) < epsilon:
        break

    U = new_U

# ==========================================================
# 🔹 6. PLOT COMBINED CONVERGENCE CURVE
# ==========================================================

plt.figure()

# GA phase
plt.plot(range(1, len(ga_history) + 1),
         ga_history,
         label="GA Phase")

# FCM phase (shifted on x-axis)
fcm_x = range(len(ga_history) + 1,
              len(ga_history) + len(fcm_history) + 1)

plt.plot(fcm_x,
         fcm_history,
         label="FCM Refinement Phase")

plt.xlabel("Iteration")
plt.ylabel("Objective Function (Jm)")
plt.title("GA + FCM Convergence Curve (# Dataset)") #Change
plt.legend()
plt.tight_layout()

output_file = "GA+FCM.jpg"  # Change
plt.savefig(output_file, format="jpg")
plt.close()

print(f"Convergence curve saved as: {output_file}")
