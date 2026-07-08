# ==========================================================
# K-MEANS CLUSTERING + 5 VALIDITY INDICES
# (Editable dataset path)
# ==========================================================

import numpy as np
import pandas as pd
import time  # <--- Added for execution time tracking
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from scipy.spatial.distance import cdist
from itertools import combinations

# Start the execution timer
start_time = time.time()

# ==========================================================
# 🔹 1. CHANGE YOUR DATASET PATH HERE
# ==========================================================
file_path = "/Users/souviksarkarjr./Downloads/Datasets/Bank.csv"   # <-- Change this path
n_clusters = 3                   # <-- Change number of clusters

# ==========================================================
# 🔹 2. LOAD DATA
# ==========================================================
df = pd.read_csv(file_path)

# Keep only numeric columns
X = df.select_dtypes(include=[np.number]).values

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================================
# 🔹 3. APPLY K-MEANS
# ==========================================================
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

# ==========================================================
# 🔹 4. VALIDITY INDICES
# ==========================================================

# 1️⃣ Silhouette Score (Higher = Better)
s_score = silhouette_score(X_scaled, labels)

# 2️⃣ Calinski-Harabasz Index (Higher = Better)
ch_index = calinski_harabasz_score(X_scaled, labels)

# 3️⃣ Davies-Bouldin Index (Lower = Better)
db_index = davies_bouldin_score(X_scaled, labels)

# ----------------------------------------------------------
# 4️⃣ Xie-Beni Index (Lower = Better)
# ----------------------------------------------------------
def xie_beni_index(X, labels, centroids):
    n = X.shape[0]
    k = centroids.shape[0]
    
    intra_dist = 0
    for i in range(k):
        cluster_points = X[labels == i]
        intra_dist += np.sum((cluster_points - centroids[i])**2)
    
    centroid_distances = cdist(centroids, centroids)
    np.fill_diagonal(centroid_distances, np.inf)
    min_centroid_dist = np.min(centroid_distances)
    
    xb = intra_dist / (n * min_centroid_dist**2)
    return xb

xb_score = xie_beni_index(X_scaled, labels, centroids)

# ----------------------------------------------------------
# 5️⃣ Dunn Index (Higher = Better)
# ----------------------------------------------------------
def dunn_index(X, labels):
    clusters = np.unique(labels)
    
    # Inter-cluster distances
    min_intercluster = np.inf
    for (i, j) in combinations(clusters, 2):
        points_i = X[labels == i]
        points_j = X[labels == j]
        dist = cdist(points_i, points_j)
        min_intercluster = min(min_intercluster, np.min(dist))
    
    # Intra-cluster distances
    max_intracluster = 0
    for i in clusters:
        points = X[labels == i]
        if len(points) > 1:
            dist = cdist(points, points)
            max_intracluster = max(max_intracluster, np.max(dist))
    
    return min_intercluster / max_intracluster

dunn = dunn_index(X_scaled, labels)

# Calculate total time taken
end_time = time.time()
execution_time = end_time - start_time

# ==========================================================
# 🔹 5. PRINT RESULTS
# ==========================================================
print("\n================= CLUSTERING RESULTS =================")
print(f"Number of Clusters: {n_clusters}")
print("------------------------------------------------------")
print(f"Silhouette Score (S-Score): {s_score:.4f}  (Higher = Better)")
print(f"Xie-Beni Index (XB-Score):  {xb_score:.4f}  (Lower = Better)")
print(f"Calinski-Harabasz Index:    {ch_index:.4f}  (Higher = Better)")
print(f"Dunn Index:                 {dunn:.4f}  (Higher = Better)")
print(f"Davies-Bouldin Index:       {db_index:.4f}  (Lower = Better)")
print(f"Execution Time:             {execution_time:.4f} seconds")  # <--- Added output
print("======================================================\n")