import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

from scipy.spatial.distance import cdist

# ==========================
# PARAMETERS
# ==========================
csv_path = "/Users/souviksarkarjr./Downloads/Experimental Dataset/Iris.csv"

n_clusters = 3
pop_size = 30
max_iter = 100
beta = 2

random.seed(42)
np.random.seed(42)

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv(csv_path)

if 'Id' in df.columns:
    df = df.drop(columns=['Id'])
elif 'id' in df.columns:
    df = df.drop(columns=['id'])

df = df.select_dtypes(include=[np.number])

if df.shape[1] == 0:
    raise ValueError("Dataset has no numeric columns!")

data = df.values

# ==========================
# NORMALIZATION
# ==========================
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# ==========================
# HELPER FUNCTIONS
# ==========================
def assign_clusters(data, centroids):
    distances = cdist(data, centroids, metric='euclidean')
    return np.argmin(distances, axis=1)


def clustering_fitness(data, centroids):
    labels = assign_clusters(data, centroids)

    fitness = 0

    for i in range(len(centroids)):
        cluster_points = data[labels == i]

        if len(cluster_points) > 0:
            fitness += np.sum((cluster_points - centroids[i]) ** 2)

    return fitness


def centroid_separation(centroids):
    """
    Larger separation is better.
    """
    dist = cdist(centroids, centroids)

    np.fill_diagonal(dist, np.inf)

    return np.min(dist)


def dominates(a1, a2, b1, b2):
    """
    Pareto dominance:
    both objectives minimized.
    """
    return (
        (a1 <= b1 and a2 <= b2)
        and
        (a1 < b1 or a2 < b2)
    )


def dunn_index(data, labels):

    unique_clusters = np.unique(labels)

    if len(unique_clusters) < 2:
        return 0

    clusters = [data[labels == k] for k in unique_clusters]

    inter_cluster = np.inf

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):

            dist = np.min(
                cdist(clusters[i], clusters[j])
            )

            inter_cluster = min(inter_cluster, dist)

    intra_cluster = 0

    for cluster in clusters:

        if len(cluster) > 1:

            dist = np.max(
                cdist(cluster, cluster)
            )

            intra_cluster = max(intra_cluster, dist)

    if intra_cluster == 0:
        return 0

    return inter_cluster / intra_cluster


def xie_beni_index(data, labels, centroids):

    N = len(data)
    K = len(centroids)

    numerator = 0

    for i in range(K):

        cluster_points = data[labels == i]

        if len(cluster_points) > 0:
            numerator += np.sum(
                (cluster_points - centroids[i]) ** 2
            )

    centroid_dist = cdist(centroids, centroids)

    np.fill_diagonal(
        centroid_dist,
        np.inf
    )

    min_centroid_dist = np.min(
        centroid_dist
    )

    if min_centroid_dist == 0:
        return 0

    return numerator / (
        N * (min_centroid_dist ** 2)
    )


# ==========================
# SEARCH SPACE
# ==========================
dim = data.shape[1]

lb = np.min(data, axis=0)
ub = np.max(data, axis=0)

# ==========================
# MULTI-OBJECTIVE LEARNING
# INITIALIZATION
# ==========================
population = [
    np.random.uniform(
        lb,
        ub,
        (n_clusters, dim)
    )
    for _ in range(pop_size)
]

obj1 = [
    clustering_fitness(data, p)
    for p in population
]

# minimize negative separation
obj2 = [
    -centroid_separation(p)
    for p in population
]

pareto_rank = []

for i in range(pop_size):

    rank = 0

    for j in range(pop_size):

        if dominates(
            obj1[j],
            obj2[j],
            obj1[i],
            obj2[i]
        ):
            rank += 1

    pareto_rank.append(rank)

sorted_idx = np.argsort(pareto_rank)

population = [
    population[i]
    for i in sorted_idx
]

fitness = [
    obj1[i]
    for i in sorted_idx
]

best_idx = np.argmin(fitness)

best_centroids = (
    population[best_idx].copy()
)

best_score = (
    fitness[best_idx]
)

convergence_curve = []

# ==========================
# FGO OPTIMIZATION LOOP
# ==========================
for t in range(max_iter):

    R = (
        1 - t / max_iter
    ) ** beta

    print(
        f"Iteration {t+1}/{max_iter} | "
        f"Best Fitness: {best_score:.6f}"
    )

    for i in range(pop_size):

        Xi = population[i].copy()

        j, k = np.random.choice(
            pop_size,
            2,
            replace=False
        )

        new_solution = (
            Xi
            +
            R
            *
            np.random.rand()
            *
            (
                population[j]
                -
                population[k]
            )
        )

        new_solution = np.clip(
            new_solution,
            lb,
            ub
        )

        # ----------------------
        # MULTI OBJECTIVES
        # ----------------------
        new_obj1 = clustering_fitness(
            data,
            new_solution
        )

        new_obj2 = -centroid_separation(
            new_solution
        )

        old_obj1 = clustering_fitness(
            data,
            population[i]
        )

        old_obj2 = -centroid_separation(
            population[i]
        )

        # Pareto-based selection
        if dominates(
            new_obj1,
            new_obj2,
            old_obj1,
            old_obj2
        ):

            population[i] = new_solution
            fitness[i] = new_obj1

        elif (
            not dominates(
                old_obj1,
                old_obj2,
                new_obj1,
                new_obj2
            )
            and
            random.random() < 0.5
        ):

            population[i] = new_solution
            fitness[i] = new_obj1

        if fitness[i] < best_score:

            best_score = fitness[i]

            best_centroids = (
                population[i].copy()
            )

    convergence_curve.append(
        best_score
    )

# ==========================
# FINAL CLUSTERING
# ==========================
labels = assign_clusters(
    data,
    best_centroids
)

# ==========================
# VALIDITY INDICES
# ==========================
s_score = silhouette_score(
    data,
    labels
)

ch_score = calinski_harabasz_score(
    data,
    labels
)

db_score = davies_bouldin_score(
    data,
    labels
)

dunn = dunn_index(
    data,
    labels
)

xb = xie_beni_index(
    data,
    labels,
    best_centroids
)

# ==========================
# RESULTS
# ==========================
print("\n========== RESULTS (FGO + Multi-Objective Learning) ==========")

print(
    "Best Fitness:",
    best_score
)

print(
    "Silhouette Score (S):",
    s_score
)

print(
    "Calinski-Harabasz (CH):",
    ch_score
)

print(
    "Dunn Index:",
    dunn
)

print(
    "Davies-Bouldin (DB):",
    db_score
)

print(
    "Xie-Beni (XB):",
    xb
)