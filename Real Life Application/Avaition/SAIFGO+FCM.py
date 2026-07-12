import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import os
import math
import random
import skfuzzy as fuzz
import pyproj

from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import cdist

base_dir = os.path.dirname(os.path.abspath(__file__))

flight_path = os.path.join(base_dir, "Dataset", "Flight.csv")
city_path = os.path.join(base_dir, "Dataset", "City.csv")

# Dataset
flights = pd.read_csv(flight_path)
geo = pd.read_csv(city_path)

# Geo Dataset
geo["city"] = geo["Location"].str.replace(" Latitude and Longitude", "", regex=False)
geo = geo[["city", "Latitude", "Longitude"]]

geo["city"] = geo["city"].str.strip().str.lower()
flights["origin"] = flights["origin"].str.strip().str.lower()
flights["destination"] = flights["destination"].str.strip().str.lower()

# Merge Coordinates
flights = flights.merge(geo, left_on="origin", right_on="city", how="left")
flights = flights.rename(columns={"Latitude": "origin_lat", "Longitude": "origin_lon"}).drop(columns=["city"])

flights = flights.merge(geo, left_on="destination", right_on="city", how="left")
flights = flights.rename(columns={"Latitude": "dest_lat", "Longitude": "dest_lon"}).drop(columns=["city"])

flights = flights.dropna(subset=["origin_lat", "origin_lon", "dest_lat", "dest_lon"])

points = pd.concat([
    flights[["origin_lat", "origin_lon"]].rename(columns={"origin_lat": "lat", "origin_lon": "lon"}),
    flights[["dest_lat", "dest_lon"]].rename(columns={"dest_lat": "lat", "dest_lon": "lon"})
]).reset_index(drop=True)

data = points[["lat", "lon"]].values

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Parameters
n_clusters = 6
pop_size = 20
max_iter = 30
beta = 2
alpha = 0.01
r_max = 1.0            
r_min = 0.0
epsilon = 1e-4         

random.seed(42)
np.random.seed(42)

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
    numerator = np.sum((U.T ** m) * dist_sq)
    
    center_dist_sq = cdist(centers, centers, metric='sqeuclidean')
    np.fill_diagonal(center_dist_sq, np.inf) 
    min_center_dist = np.min(center_dist_sq)
    
    denominator = N * min_center_dist
    return numerator / denominator if denominator > 0 else np.inf

# SAIFGO Initialization
N, dim = data.shape
D = n_clusters * dim
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

raw_pool.sort(key=clustering_fitness)
population = np.array(raw_pool[:pop_size])
fitness = np.array([clustering_fitness(p) for p in population])

best_idx = np.argmin(fitness)
best_Z = population[best_idx].copy()
best_score = fitness[best_idx]

strategy_success = np.ones(3)

# SAIFGO Optimization
print(f"\n--- Starting SAIFGO Optimization (Total Iterations: {max_iter}) ---")
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
print("\n--- Refining with Fuzzy C-Means (FCM) ---")
data_T = data.T
best_centroids = best_Z.reshape(n_clusters, dim)

distances = cdist(data, best_centroids)
distances = np.fmax(distances, 1e-10)
temp = distances ** 2.0 
U_init = (1.0 / temp) / np.sum((1.0 / temp), axis=1, keepdims=True)
U_init = U_init.T

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data_T,
    c=n_clusters,
    m=2,
    error=0.005,
    maxiter=500,
    init=U_init
)

labels = np.argmax(u, axis=0)
centers = scaler.inverse_transform(cntr)
points["cluster"] = labels

# --- Clustering Validity Indices Calculation (MEMORY SAFE) ---
print("\nCalculating metrics... (this may take a moment)")

sample_size = min(5000, len(data))
sample_indices = np.random.choice(len(data), size=sample_size, replace=False)

data_sample = data[sample_indices]
labels_sample = labels[sample_indices]

S_score = silhouette_score(data_sample, labels_sample)
DB_score = davies_bouldin_score(data_sample, labels_sample)
CH_score = calinski_harabasz_score(data_sample, labels_sample)
Dunn_score = dunn_index(data_sample, labels_sample)
XB_score = xie_beni_index(data, u, cntr, m=2) 

print("\n--- Clustering Validity Indices ---")
print(f"Fuzzy Partition Coefficient (FPC) : {fpc:.4f}")
print(f"Silhouette Index (S)            : {S_score:.4f}")
print(f"Davies-Bouldin Index (DB)       : {DB_score:.4f}")
print(f"Calinski-Harabasz Index (CH)    : {CH_score:.4f}")
print(f"Dunn Index                      : {Dunn_score:.4f}")
print(f"Xie-Beni Index (XB)             : {XB_score:.4f}")
print("-----------------------------------\n")

# Geo-Frame
print("Generating and saving plot...")
gdf = gpd.GeoDataFrame(
    points,
    geometry=gpd.points_from_xy(points["lon"], points["lat"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

centers_gdf = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(centers[:,1], centers[:,0]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

# --- AESTHETIC PLOTTING ---
plt.style.use("default")
fig, ax = plt.subplots(figsize=(16, 16), dpi=150)

# Memory-optimized plotting using PyProj
transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
for _, row in flights.iterrows():
    x_orig, y_orig = transformer.transform(row["origin_lon"], row["origin_lat"])
    x_dest, y_dest = transformer.transform(row["dest_lon"], row["dest_lat"])
    ax.plot([x_orig, x_dest], [y_orig, y_dest], color="#3b5b92", alpha=0.15, linewidth=0.6, zorder=1)

# Plot Points (Airports)
cmap_colors = plt.cm.tab10.colors
gdf.plot(ax=ax, column="cluster", categorical=True, cmap="tab10", 
         markersize=60, edgecolor="black", linewidth=0.8, zorder=2)

# Plot Centers
centers_gdf.plot(ax=ax, color="#d62728", markersize=350,
                 marker="X", edgecolor="black", linewidth=1.5, zorder=3)

# Add Map Background
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=1.0)

# Remove axes ticks but keep the border frame
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_edgecolor('gray')
    spine.set_linewidth(1.5)

# Add Titles and Captions
ax.set_title("Domestic Flight Network and Airport Across India (using SAIFGO+FCM)", 
             fontsize=20, fontweight="bold", pad=15)
fig.text(0.5, 0.05, "(a) Aviation Network Using SAIFGO+FCM", 
         ha="center", fontsize=16, fontweight="bold")

# Custom Professional Legend
legend_elements = []
unique_clusters = np.sort(points["cluster"].unique())

for cluster_id in unique_clusters:
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', label=f'Cluster {cluster_id}',
               markerfacecolor=cmap_colors[cluster_id % 10], markeredgecolor='black', markersize=10)
    )

legend_elements.append(
    Line2D([0], [0], marker='X', color='w', label='Cluster Center',
           markerfacecolor='#d62728', markeredgecolor='black', markersize=12, markeredgewidth=1.5)
)
legend_elements.append(
    Line2D([0], [0], color='#3b5b92', lw=1.5, label='Flight Route')
)

ax.legend(handles=legend_elements, loc='lower left', title="Legend", 
          fontsize=12, title_fontsize=14, framealpha=0.9, edgecolor="gray")

# Adjust layout to fit caption
plt.tight_layout(rect=[0, 0.08, 1, 1])

# Auto Save
output_dir = os.path.join(base_dir, "Results")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "SAIFGO_FCM.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')

print(f"Done! Plot saved at: {output_path}")
plt.show()