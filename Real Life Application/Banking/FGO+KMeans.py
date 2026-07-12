import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist

# FGO PARAMETERS

csv_path = "/Users/souviksarkarjr./Downloads/Real Life Application/Banking/Dataset/creditcard.csv"

n_clusters = 4
pop_size = 20
max_iter = 30

sample_size = 10000

save_cluster_plot = "cluster_plot_FGO+KMeans.png"

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

dim = data.shape[1]
D = n_clusters * dim

# Helper Functions

def clustering_fitness(Z):
    centroids = Z.reshape(n_clusters, dim)
    distances = cdist(data, centroids)
    labels = np.argmin(distances, axis=1)
    fitness = 0
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            fitness += np.sum((cluster_points - centroids[i])**2)
    return fitness

def dunn_index(X, labels):
    """Computes the Dunn Index"""
    min_inter_dist = np.inf
    max_intra_dist = 0
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    
    for i in range(n_labels):
        cluster_i = X[labels == unique_labels[i]]
        if len(cluster_i) == 0: continue
        
        # Max intra-cluster distance
        intra_dists = cdist(cluster_i, cluster_i)
        max_intra_dist = max(max_intra_dist, np.max(intra_dists))
        
        # Min inter-cluster distance
        for j in range(i + 1, n_labels):
            cluster_j = X[labels == unique_labels[j]]
            if len(cluster_j) == 0: continue
            inter_dists = cdist(cluster_i, cluster_j)
            min_inter_dist = min(min_inter_dist, np.min(inter_dists))
            
    if max_intra_dist == 0:
        return 0
    return min_inter_dist / max_intra_dist

def xie_beni_index(X, labels, centroids):
    """Computes the Xie-Beni Index"""
    n = len(X)
    sse = 0
    
    # Calculate Total Within-Cluster Variation
    for i, center in enumerate(centroids):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            sse += np.sum((cluster_points - center) ** 2)
            
    # Calculate Minimum Distance Between Centroids
    centroid_dists = cdist(centroids, centroids, metric='sqeuclidean')
    np.fill_diagonal(centroid_dists, np.inf)
    min_centroid_dist = np.min(centroid_dists)
    
    if min_centroid_dist == 0:
        return np.inf
    return sse / (n * min_centroid_dist)

# FGO Initialization

lb_base = np.min(data, axis=0)
ub_base = np.max(data, axis=0)
LB = np.tile(lb_base, n_clusters)
UB = np.tile(ub_base, n_clusters)

raw_pool = []
r_chaotic = random.random()

for _ in range(pop_size):
    Z_primary = np.zeros(D)
    for m in range(D):
        r_chaotic = 4.0 * r_chaotic * (1.0 - r_chaotic)
        Z_primary[m] = LB[m] + r_chaotic * (UB[m] - LB[m])
        
    raw_pool.append(Z_primary)
    raw_pool.append(LB + UB - Z_primary)

# Sort to minimize fitness (SSE)
raw_pool.sort(key=clustering_fitness)
population = np.array(raw_pool[:pop_size])
fitness = np.array([clustering_fitness(p) for p in population])

best_idx = np.argmin(fitness)
best_Z = population[best_idx].copy()
best_score = fitness[best_idx]

print("\n--- Starting FGO Optimization ---\n")

# FGO Optimization

for t in range(max_iter):
    
    for i in range(pop_size):
        Xi = population[i].copy()
        
        # 1. Spore Dispersal (Exploration)
        # Search radius linearly decreases to encourage exploitation in later iterations
        radius = 0.2 * (1.0 - t / max_iter) 
        spore = Xi + (np.random.rand(D) - 0.5) * 2.0 * radius * (UB - LB)
        spore = np.clip(spore, LB, UB)
        
        # 2. Hyphal Growth (Exploitation towards the best known solution)
        hypha = Xi + np.random.rand(D) * (best_Z - Xi)
        hypha = np.clip(hypha, LB, UB)
        
        # Evaluate fitness for the newly generated structures
        fit_spore = clustering_fitness(spore)
        fit_hypha = clustering_fitness(hypha)
        
        # Survival of the fittest selection among parent, spore, and hypha
        best_new_fit = fitness[i]
        best_new_sol = Xi
        
        if fit_spore < best_new_fit:
            best_new_fit = fit_spore
            best_new_sol = spore
            
        if fit_hypha < best_new_fit:
            best_new_fit = fit_hypha
            best_new_sol = hypha
            
        # Update population if a better structure was found
        if best_new_fit < fitness[i]:
            population[i] = best_new_sol
            fitness[i] = best_new_fit
            
            # Update global best
            if best_new_fit < best_score:
                best_score = best_new_fit
                best_Z = best_new_sol.copy()
                
    print(f"Iteration {t+1:03d}/{max_iter}  Best Fitness = {best_score:.4f}")

# K-Means 

print("\nRefining clusters using K-Means")

initial_centroids = best_Z.reshape(n_clusters, dim)

kmeans = KMeans(
    n_clusters=n_clusters,
    init=initial_centroids,
    n_init=1,
    max_iter=300,
    random_state=42
)

labels = kmeans.fit_predict(data)
final_centroids = kmeans.cluster_centers_

# --- Clustering Validity Indices Evaluation ---
print("\nCalculating Clustering Validity Indices...")

s_idx = silhouette_score(data, labels)
ch_idx = calinski_harabasz_score(data, labels)
db_idx = davies_bouldin_score(data, labels)
dunn_idx = dunn_index(data, labels)
xb_idx = xie_beni_index(data, labels, final_centroids)

print(f"Silhouette Score (S)        : {s_idx:.4f} (Higher is better)")
print(f"Calinski-Harabasz (CH)      : {ch_idx:.4f} (Higher is better)")
print(f"Davies-Bouldin (DB)         : {db_idx:.4f} (Lower is better)")
print(f"Dunn Index                  : {dunn_idx:.6f} (Higher is better)")
print(f"Xie-Beni Index (XB)         : {xb_idx:.6f} (Lower is better)")
print("-" * 50)

# Result Auto-save

pd.DataFrame(final_centroids).to_csv("centroids.csv", index=False)
pd.DataFrame(labels, columns=["cluster"]).to_csv("cluster_labels.csv", index=False)

print("\nCentroids and labels saved")

# PCA

pca = PCA(n_components=3)
data_pca = pca.fit_transform(data)

# 3D Plot Matched to Reference Image

plt.style.use("seaborn-v0_8-whitegrid")

# Create figure with the grey border outline seen in the reference
fig = plt.figure(figsize=(13, 10), linewidth=2, edgecolor='gray')
fig.patch.set_facecolor('white')

ax = fig.add_subplot(111, projection='3d')

# Data points only - removed edgecolors, adjusted size
scatter = ax.scatter(
    data_pca[:,0],
    data_pca[:,1],
    data_pca[:,2],
    c=labels,
    cmap="viridis",
    s=12,
    alpha=0.9
)

ax.set_xlabel("Principal Component 1", fontsize=12, labelpad=10)
ax.set_ylabel("Principal Component 2", fontsize=12, labelpad=10)
ax.set_zlabel("Principal Component 3", fontsize=12, labelpad=10)

ax.set_title(
    "Clustering Visualization (FGO + K-Means)",
    fontsize=20,
    pad=30
)

# Shrunk colorbar to match reference scale
cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.08)
cbar.set_label("Cluster Assignment", fontsize=13, labelpad=10)

# Viewing angle mapping the reference
ax.view_init(elev=20, azim=135)

# Add sub-caption text at the bottom center
fig.text(
    0.5, 0.05, 
    "(a) Credit Card Anomaly Detection using FGO+KMeans", 
    ha="center", 
    va="center", 
    fontsize=14, 
    fontweight="bold"
)

# Adjust plot layout to avoid overlapping with the bottom text

plt.subplots_adjust(bottom=0.15)

# Ensure border properties are saved
plt.savefig(
    save_cluster_plot, 
    dpi=600, 
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
    edgecolor=fig.get_edgecolor()
)

plt.show()

print("Plots saved successfully")