import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# SAIFGO PARAMETERS

csv_path = "/Users/souviksarkarjr./Documents/IFGO/Real Life Application/Banking/Dataset/creditcard.csv"

n_clusters = 5
pop_size = 30
max_iter = 100

beta = 2
alpha = 0.01

sample_size = 10000

save_cluster_plot = "cluster_plot_IFGO+KMeans.png"
save_convergence_plot = "convergence_plot_IFGO+KMeans.png"

random.seed(42)
np.random.seed(42)

# Data Loading

print("Loading dataset...")

df = pd.read_csv(csv_path)

if "Class" in df.columns:
    labels_true = df["Class"]
    df = df.drop(columns=["Class"])

if len(df) > sample_size:
    df = df.sample(sample_size, random_state=42)

data = df.values

print("Dataset used:", data.shape)

# Normalization

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Helper Function

def assign_clusters(data, centroids):
    distances = cdist(data, centroids)
    return np.argmin(distances, axis=1)


def clustering_fitness(data, centroids):

    labels = assign_clusters(data, centroids)

    fitness = 0

    for i in range(len(centroids)):

        cluster_points = data[labels == i]

        if len(cluster_points) > 0:
            fitness += np.sum((cluster_points - centroids[i])**2)

    return fitness


def levy_flight(dim, lam=1.5):

    sigma = (math.gamma(1 + lam) * np.sin(np.pi * lam / 2) /
             (math.gamma((1 + lam) / 2) * lam *
              2 ** ((lam - 1) / 2))) ** (1 / lam)

    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)

    return u / (np.abs(v) ** (1 / lam))

# Initialization

dim = data.shape[1]

lb = np.min(data, axis=0)
ub = np.max(data, axis=0)

population = []

for _ in range(pop_size):

    centroids = np.random.uniform(lb, ub, (n_clusters, dim))
    opposite = lb + ub - centroids

    population.append(centroids)
    population.append(opposite)

population = population[:pop_size]

fitness = [clustering_fitness(data, p) for p in population]

best_idx = np.argmin(fitness)

best_centroids = population[best_idx].copy()
best_score = fitness[best_idx]

strategy_success = np.ones(3)

convergence_curve = []

print("\nStarting IFGO optimization\n")

# Optimization

for t in range(max_iter):

    R = (1 - t / max_iter) ** beta

    probs = strategy_success / np.sum(strategy_success)

    elite_count = max(1, int(0.1 * pop_size))

    elite_idx = np.argsort(fitness)[:elite_count]

    elite_mean = np.mean([population[i] for i in elite_idx], axis=0)

    for i in range(pop_size):

        strategy = np.random.choice(3, p=probs)

        Xi = population[i].copy()

        if strategy == 0:

            new_solution = Xi + R * np.random.rand() * (elite_mean - Xi)

        elif strategy == 1:

            j, k = np.random.choice(pop_size, 2, replace=False)

            new_solution = Xi + R * np.random.rand() * (population[j] - population[k])

        else:

            new_solution = Xi + alpha * levy_flight(dim).reshape(1, -1)

        new_solution = np.clip(new_solution, lb, ub)

        new_fit = clustering_fitness(data, new_solution)

        if new_fit < fitness[i]:

            population[i] = new_solution
            fitness[i] = new_fit

            strategy_success[strategy] += 1

            if new_fit < best_score:

                best_score = new_fit
                best_centroids = new_solution.copy()

    convergence_curve.append(best_score)

    print(f"Iteration {t+1}/{max_iter}  Best Fitness = {best_score}")

# K-Means 

print("\nRefining clusters using K-Means")

kmeans = KMeans(
    n_clusters=n_clusters,
    init=best_centroids,
    n_init=1,
    max_iter=300,
    random_state=42
)

labels = kmeans.fit_predict(data)
final_centroids = kmeans.cluster_centers_

# Result Auto-save

pd.DataFrame(final_centroids).to_csv("centroids.csv", index=False)
pd.DataFrame(labels, columns=["cluster"]).to_csv("cluster_labels.csv", index=False)

print("Centroids and labels saved")

# CONVERGENCE PLOT

plt.figure()

plt.plot(convergence_curve)

plt.title("IFGO Optimization Convergence")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")

plt.savefig(save_convergence_plot, dpi=300)

# PCA

pca = PCA(n_components=3)

data_pca = pca.fit_transform(data)
centroids_pca = pca.transform(final_centroids)

# 3D Plot

plt.style.use("seaborn-v0_8-whitegrid")

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    data_pca[:,0],
    data_pca[:,1],
    data_pca[:,2],
    c=labels,
    cmap="viridis",
    s=18,
    alpha=0.75,
    edgecolors="black",
    linewidths=0.15
)

# Centroids

ax.scatter(
    centroids_pca[:,0],
    centroids_pca[:,1],
    centroids_pca[:,2],
    c="red",
    s=700,
    marker="X",
    edgecolors="black",
    linewidths=2.5,
    depthshade=False,
    label="Cluster Centroids"
)

ax.set_xlabel("Principal Component 1", fontsize=13)
ax.set_ylabel("Principal Component 2", fontsize=13)
ax.set_zlabel("Principal Component 3", fontsize=13)

ax.set_title(
    "Clustering Visualization (SAIFGO + K-Means)",
    fontsize=16,
    pad=20
)

cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("Cluster Assignment", fontsize=12)

ax.legend(loc="upper left")

ax.view_init(elev=25, azim=135)

plt.tight_layout()

plt.savefig(save_cluster_plot, dpi=600, bbox_inches="tight")

plt.show()

print("Plots saved successfully")