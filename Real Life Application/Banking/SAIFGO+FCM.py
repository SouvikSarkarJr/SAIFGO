import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import cdist

# SAIFGO PARAMETERS

csv_path = "/Users/souviksarkarjr./Downloads/Real Life Application/Banking/Dataset/creditcard.csv"

n_clusters = 4
pop_size = 20
max_iter = 30

beta = 2
alpha = 0.01
r_max = 1.0            
r_min = 0.0
epsilon = 1e-4

fcm_m = 2
fcm_iter = 100

sample_size = 10000

save_cluster_plot = "cluster_plot_SAIFGO+FCM.png"

random.seed(42)
np.random.seed(42)

# Loading of Data

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

def levy_flight(dim_size, lam=1.5):
    sigma = (math.gamma(1 + lam) * np.sin(np.pi * lam / 2) /
             (math.gamma((1 + lam) / 2) * lam *
              2 ** ((lam - 1) / 2))) ** (1 / lam)
    u = np.random.normal(0, sigma, dim_size)
    v = np.random.normal(0, 1, dim_size)
    return u / (np.abs(v) ** (1 / lam))

def dunn_index(X, labels):
    distances = cdist(X, X)
    clusters = np.unique(labels)
    
    max_intra = 0
    for c in clusters:
        idx = np.where(labels == c)[0]
        if len(idx) > 1:
            max_intra = max(max_intra, np.max(distances[idx][:, idx]))
            
    min_inter = np.inf
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            idx_i = np.where(labels == clusters[i])[0]
            idx_j = np.where(labels == clusters[j])[0]
            if len(idx_i) > 0 and len(idx_j) > 0:
                min_inter = min(min_inter, np.min(distances[idx_i][:, idx_j]))
                
    return min_inter / max_intra if max_intra > 0 else 0

def xie_beni_index(X, U, centers, m=2):
    N = X.shape[0]
    
    dist_sq = cdist(X, centers, metric='sqeuclidean') 
    numerator = np.sum((U ** m) * dist_sq)
    
    center_dist_sq = cdist(centers, centers, metric='sqeuclidean')
    np.fill_diagonal(center_dist_sq, np.inf) 
    min_center_dist = np.min(center_dist_sq)
    
    denominator = N * min_center_dist
    return numerator / denominator if denominator > 0 else np.inf

# SAIFGO Initialization

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

strategy_success = np.ones(3)

print("\nStarting SAIFGO optimization\n")

# SAIFGO Optimization

for t in range(max_iter):
    r_t = r_max - ((t / max_iter) ** beta) * (r_max - r_min)
    probs = strategy_success / np.sum(strategy_success)
    
    elite_count = max(1, int(0.1 * pop_size))
    elite_idx = np.argsort(fitness)[:elite_count] 
    elite_mean = np.mean(population[elite_idx], axis=0)
    
    strategies = np.random.choice(3, size=pop_size, p=probs)
    
    for i in range(pop_size):
        strategy = strategies[i]
        Xi = population[i].copy()
        
        if strategy == 0:
            new_solution = Xi + r_t * (elite_mean - Xi)
        elif strategy == 1:
            a, b = np.random.choice([idx for idx in range(pop_size) if idx != i], 2, replace=False)
            new_solution = Xi + r_t * (population[a] - population[b])
        else:
            new_solution = Xi + alpha * levy_flight(D)
            
        new_solution = np.clip(new_solution, LB, UB)
        new_fit = clustering_fitness(new_solution)
        
        if new_fit < fitness[i]:
            population[i] = new_solution
            fitness[i] = new_fit
            strategy_success[strategy] += 1.0
            
            if new_fit < best_score:
                best_score = new_fit
                best_Z = new_solution.copy()
                
    strategy_success = np.maximum(strategy_success * 0.9, epsilon)
    
    print(f"Iteration {t+1:02d}/{max_iter} | Best Fitness (SSE): {best_score:.4f}")

# FCM

def fuzzy_c_means(data, initial_centroids, m=2):
    centroids = initial_centroids.copy()
    U = None
    for _ in range(fcm_iter):
        dist = cdist(data, centroids)
        dist = np.fmax(dist, 1e-10)
        power = 2/(m-1)
        inv_dist = dist ** (-power)
        U = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)
        um = U ** m
        new_centroids = (um.T @ data) / np.sum(um.T, axis=1, keepdims=True)
        if np.linalg.norm(new_centroids - centroids) < 1e-5:
            break
        centroids = new_centroids
    labels = np.argmax(U, axis=1)
    return centroids, labels, U

print("\nRefining clusters using FCM")

final_centroids, labels, U = fuzzy_c_means(data, best_Z.reshape(n_clusters, dim), m=fcm_m)

print("Number of clusters formed:", len(np.unique(labels)))

# --- Clustering Validity Indices Calculation ---
print("\nCalculating Validity Indices (this might take a moment)...")
S_score = silhouette_score(data, labels)
DB_score = davies_bouldin_score(data, labels)
CH_score = calinski_harabasz_score(data, labels)
Dunn_score = dunn_index(data, labels)
XB_score = xie_beni_index(data, U, final_centroids, m=fcm_m)

print("\n--- Clustering Validity Indices ---")
print(f"Silhouette Index (S)         : {S_score:.4f} (Higher is better)")
print(f"Davies-Bouldin Index (DB)    : {DB_score:.4f} (Lower is better)")
print(f"Calinski-Harabasz Index (CH) : {CH_score:.4f} (Higher is better)")
print(f"Dunn Index                   : {Dunn_score:.4f} (Higher is better)")
print(f"Xie-Beni Index (XB)          : {XB_score:.4f} (Lower is better)")
print("-----------------------------------\n")

# Result Auto-save

pd.DataFrame(final_centroids).to_csv("centroids.csv", index=False)
pd.DataFrame(labels, columns=["cluster"]).to_csv("cluster_labels.csv", index=False)

print("Centroids and labels saved")


# PCA

pca = PCA(n_components=3)

data_pca = pca.fit_transform(data)

# 3D Plotting - Updated to match reference aesthetic

plt.style.use("seaborn-v0_8-whitegrid")

# Create figure with gray edge border
fig = plt.figure(figsize=(12, 9), dpi=150, edgecolor='gray', linewidth=2)
ax = fig.add_subplot(111, projection='3d')

# Data points (Removed edgecolors, adjusted size/alpha)
scatter = ax.scatter(
    data_pca[:, 0],
    data_pca[:, 1],
    data_pca[:, 2],
    c=labels,
    cmap="viridis",
    s=15,
    alpha=0.75,
    edgecolors="none" 
)

# Axis Labels
ax.set_xlabel("Principal Component 1", fontsize=12, labelpad=10)
ax.set_ylabel("Principal Component 2", fontsize=12, labelpad=10)
ax.set_zlabel("Principal Component 3", fontsize=12, labelpad=10)

# Titles and Captions
ax.set_title("Clustering Visualization (SAIFGO + FCM)", fontsize=18, pad=25)

fig.text(0.5, 0.04, "(a) Credit Card Anomaly Detection using SAIFGO+FCM", 
         ha="center", fontsize=12, fontweight="bold")

# Colorbar adjustments
cbar = plt.colorbar(scatter, ax=ax, shrink=0.55, aspect=20, pad=0.1)
cbar.set_label("Cluster Assignment", fontsize=12)

# Viewing Angle
ax.view_init(elev=20, azim=135)

# Layout adjustments to make room for the caption at the bottom
plt.subplots_adjust(bottom=0.15) 

plt.savefig(save_cluster_plot, dpi=300, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())

plt.show()

print("Plot saved successfully")