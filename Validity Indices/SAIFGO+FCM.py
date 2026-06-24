import numpy as np
import pandas as pd
import math
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist

csv_path = "/Users/souvik/Documents/Project-II/Data-sets/Wave.csv" # CSV file path
n_clusters = 3
pop_size = 30
max_iter = 100
beta = 2
alpha = 0.01
fcm_m = 2
fcm_iter = 100

random.seed(42)
np.random.seed(42)

# Load & Process Data

df = pd.read_csv(csv_path)
df = df.select_dtypes(include=[np.number])

if df.shape[1] == 0:
    raise ValueError("Dataset has no numeric columns!")

data = df.values
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Helper Function

def assign_clusters(data, centroids):
    distances = cdist(data, centroids, metric='euclidean')
    return np.argmin(distances, axis=1)

def clustering_fitness(data, centroids):
    labels = assign_clusters(data, centroids)
    fitness = 0
    for i in range(len(centroids)):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            fitness += np.sum((cluster_points - centroids[i])**2)
    return fitness

def dunn_index(data, labels):
    unique_clusters = np.unique(labels)
    clusters = [data[labels == k] for k in unique_clusters]

    inter_cluster = np.inf
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            dist = np.min(cdist(clusters[i], clusters[j]))
            inter_cluster = min(inter_cluster, dist)

    intra_cluster = 0
    for cluster in clusters:
        if len(cluster) > 1:
            dist = np.max(cdist(cluster, cluster))
            intra_cluster = max(intra_cluster, dist)

    return inter_cluster / intra_cluster if intra_cluster != 0 else 0

def xie_beni_index(data, labels, centroids):
    N = len(data)
    K = len(centroids)

    numerator = 0
    for i in range(K):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            numerator += np.sum((cluster_points - centroids[i])**2)

    centroid_dist = cdist(centroids, centroids)
    np.fill_diagonal(centroid_dist, np.inf)
    min_centroid_dist = np.min(centroid_dist)

    return numerator / (N * min_centroid_dist) if min_centroid_dist != 0 else 0

def levy_flight(dim, lam=1.5):
    sigma = (math.gamma(1 + lam) * np.sin(np.pi * lam / 2) /
             (math.gamma((1 + lam) / 2) * lam *
              2 ** ((lam - 1) / 2))) ** (1 / lam)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    return u / (np.abs(v) ** (1 / lam))

# SAIFGO Initialization

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

# SAIFGO Optimization

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

# FCM Refinement

print("\nRefining solution using FCM...")

def fuzzy_c_means(data, initial_centroids, m=2, max_iter=100, error=1e-5):

    N = data.shape[0]
    K = initial_centroids.shape[0]
    centroids = initial_centroids.copy()

    for _ in range(max_iter):

        dist = cdist(data, centroids, metric='euclidean')
        dist = np.fmax(dist, 1e-10)

        # Update Membership

        power = 2 / (m - 1)
        temp = dist ** power
        denominator = np.sum((1 / temp), axis=1, keepdims=True)
        U = (1 / temp) / denominator

        um = U ** m
        new_centroids = (um.T @ data) / np.sum(um.T, axis=1, keepdims=True)

        if np.linalg.norm(new_centroids - centroids) < error:
            break

        centroids = new_centroids

    labels = np.argmax(U, axis=1)
    return centroids, labels

final_centroids, labels = fuzzy_c_means(data, best_centroids, fcm_m, fcm_iter)


# Clustering Validity Indices

s_score = silhouette_score(data, labels)
ch_score = calinski_harabasz_score(data, labels)
db_score = davies_bouldin_score(data, labels)
dunn = dunn_index(data, labels)
xb = xie_beni_index(data, labels, final_centroids)

print("\n========== FINAL RESULTS (AMFGO + FCM) ==========")
print("Silhouette Score (S):", s_score)
print("Calinski-Harabasz (CH):", ch_score)
print("Dunn Index:", dunn)
print("Davies-Bouldin (DB):", db_score)
print("Xie-Beni (XB):", xb)
