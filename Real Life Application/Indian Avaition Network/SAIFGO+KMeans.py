import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import os
import math
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from matplotlib.lines import Line2D

base_dir = os.path.dirname(os.path.abspath(__file__))

flight_path = os.path.join(base_dir, "Dataset", "Flight.csv")
city_path = os.path.join(base_dir, "Dataset", "City.csv")

flights = pd.read_csv(flight_path)
geo = pd.read_csv(city_path)

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
pop_size = 30
max_iter = 100
beta = 2
alpha = 0.01

random.seed(42)
np.random.seed(42)

# Helper Functions

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

# K-Means

kmeans = KMeans(
    n_clusters=n_clusters,
    init=best_centroids,
    n_init=1,
    max_iter=300,
    random_state=42
)

kmeans.fit(data)

labels = kmeans.labels_
centers = scaler.inverse_transform(kmeans.cluster_centers_)

points["cluster"] = labels

# Geo Data Frame

gdf = gpd.GeoDataFrame(
    points,
    geometry=gpd.points_from_xy(points["lon"], points["lat"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

centers_gdf = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(centers[:, 1], centers[:, 0]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

# Plot

plt.style.use("default")
fig, ax = plt.subplots(figsize=(18, 16), dpi=300)

for _, row in flights.iterrows():
    line = gpd.GeoSeries(
        gpd.points_from_xy(
            [row["origin_lon"], row["dest_lon"]],
            [row["origin_lat"], row["dest_lat"]]
        ),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    ax.plot(line.x, line.y, color="#4C72B0", alpha=0.18, linewidth=0.8)

gdf.plot(ax=ax, column="cluster", cmap="tab10", markersize=120,
         edgecolor="black", linewidth=0.7)

centers_gdf.plot(ax=ax, color="#D62728", markersize=500,
                 marker="X", edgecolor="black", linewidth=1.8)

ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.9)

ax.set_axis_off()
plt.title("SAIFGO + KMeans Clustering on Indian Aviation Network", fontsize=18)

plt.tight_layout()

# Saving Output

output_dir = os.path.join(base_dir, "Results")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "SAIFGO_KMeans.png")

plt.savefig(output_path, dpi=600, bbox_inches="tight")

print(f"Saved at: {output_path}")

plt.show()