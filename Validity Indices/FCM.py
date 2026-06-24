# ==========================================================
# FUZZY C-MEANS (FCM) + 5 VALIDITY INDICES
# (Editable dataset path)
# ==========================================================

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from scipy.spatial.distance import cdist
from itertools import combinations

# ==========================================================
# 🔹 1. CHANGE YOUR DATASET PATH HERE
# ==========================================================
file_path = "/Users/souvik/Documents/Project-II/Data-sets/Bank.csv"  #Change
n_clusters = 3         # Change number of clusters
m = 2.0                # Fuzziness coefficient (usually 2)

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
# 🔹 3. APPLY FUZZY C-MEANS
# ==========================================================

# FCM expects data in transpose form (features x samples)
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    X_scaled.T,
    c=n_clusters,
    m=m,
    error=0.005,
    maxiter=1000,
    init=None,
    seed=42
)

# Convert fuzzy memberships to crisp labels
labels = np.argmax(u, axis=0)
centroids = cntr

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
def xie_beni_index(X, u, centroids, m):
    n = X.shape[0]
    c = centroids.shape[0]
    
    dist = cdist(X, centroids, metric='euclidean') ** 2
    numerator = np.sum((u.T ** m) * dist)
    
    centroid_distances = cdist(centroids, centroids)
    np.fill_diagonal(centroid_distances, np.inf)
    min_centroid_dist = np.min(centroid_distances) ** 2
    
    xb = numerator / (n * min_centroid_dist)
    return xb

xb_score = xie_beni_index(X_scaled, u, centroids, m)

# ----------------------------------------------------------
# 5️⃣ Dunn Index (Higher = Better)
# ----------------------------------------------------------
def dunn_index(X, labels):
    clusters = np.unique(labels)
    
    min_intercluster = np.inf
    for (i, j) in combinations(clusters, 2):
        points_i = X[labels == i]
        points_j = X[labels == j]
        dist = cdist(points_i, points_j)
        min_intercluster = min(min_intercluster, np.min(dist))
    
    max_intracluster = 0
    for i in clusters:
        points = X[labels == i]
        if len(points) > 1:
            dist = cdist(points, points)
            max_intracluster = max(max_intracluster, np.max(dist))
    
    return min_intercluster / max_intracluster

dunn = dunn_index(X_scaled, labels)

# ==========================================================
# 🔹 5. PRINT RESULTS
# ==========================================================
print("\n================= FCM CLUSTERING RESULTS =================")
print(f"Number of Clusters: {n_clusters}")
print("----------------------------------------------------------")
print(f"Silhouette Score (S-Score): {s_score:.4f}  (Higher = Better)")
print(f"Xie-Beni Index (XB-Score):  {xb_score:.4f}  (Lower = Better)")
print(f"Calinski-Harabasz Index:    {ch_index:.4f}  (Higher = Better)")
print(f"Dunn Index:                 {dunn:.4f}  (Higher = Better)")
print(f"Davies-Bouldin Index:       {db_index:.4f}  (Lower = Better)")
print("==========================================================\n")
