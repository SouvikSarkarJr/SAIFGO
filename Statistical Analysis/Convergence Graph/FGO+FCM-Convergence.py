import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import os

# ==========================================================
# 🔹 1. DATASET PATH
# ==========================================================
file_path = "/Users/souvik/Documents/Project-II/Data-sets/Bank.csv" # Change
n_clusters = 3
m = 2.0
epsilon = 1e-5

# FGO parameters
n_fungi = 20
fgo_iterations = 30
growth_rate = 0.5
spore_rate = 0.3

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

def fitness(centroids):
    U = compute_membership(X, centroids, m)
    return objective_function(X, U, centroids, m)

# ==========================================================
# 🔹 4. FUNGAL GROWTH OPTIMIZATION (FGO) PHASE
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

    for i in range(n_fungi):

        # Mycelium growth toward best
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

    fgo_history.append(fitness_scores[best_idx])

# ==========================================================
# 🔹 5. FCM REFINEMENT PHASE (Manual)
# ==========================================================

U = compute_membership(X, global_best, m)
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

# FGO phase
plt.plot(range(1, len(fgo_history) + 1),
         fgo_history,
         label="FGO Phase")

# FCM refinement phase
fcm_x = range(len(fgo_history) + 1,
              len(fgo_history) + len(fcm_history) + 1)

plt.plot(fcm_x,
         fcm_history,
         label="FCM Refinement Phase")

plt.xlabel("Iteration")
plt.ylabel("Objective Function (Jm)")
plt.title("FGO + FCM Convergence Curve (Bank Dataset)") # Change
plt.legend()
plt.tight_layout()

output_file = "Bank_FGO_FCM_Convergence.jpg" # Change
plt.savefig(output_file, format="jpg")
plt.close()

print(f"Convergence curve saved as: {output_file}")