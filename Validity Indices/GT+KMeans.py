import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tqdm import tqdm

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    X = preprocessor.fit_transform(df)

    if hasattr(X, "toarray"):
        X = X.toarray()

    return X


# -----------------------------
# GENERAL TREND EXTRACTION
# -----------------------------
def compute_general_trend(X):
    """
    Extracts trend strength per feature using variance + slope idea
    """
    trends = np.zeros(X.shape[1])

    for i in tqdm(range(X.shape[1]), desc="Computing trends"):
        feature = X[:, i]

        # Trend strength = variance + linear slope magnitude
        var = np.var(feature)

        # simple slope (least squares)
        x_axis = np.arange(len(feature))
        slope = np.polyfit(x_axis, feature, 1)[0]

        trends[i] = var + abs(slope)

    # Normalize
    trends = trends / np.max(trends)

    return trends


# -----------------------------
# VALIDITY INDICES
# -----------------------------
def davies_bouldin_index(X, labels, centers):
    k = len(centers)
    clusters = [X[labels == i] for i in range(k)]

    S = np.array([
        np.mean(cdist(cluster, [centers[i]])) if len(cluster) > 0 else 0
        for i, cluster in enumerate(clusters)
    ])

    M = cdist(centers, centers)
    np.fill_diagonal(M, np.inf)

    R = (S[:, None] + S[None, :]) / M
    return np.mean(np.max(R, axis=1))


def dunn_index(X, labels):
    clusters = [X[labels == i] for i in np.unique(labels)]

    inter, intra = np.inf, 0

    for i in range(len(clusters)):
        if len(clusters[i]) > 1:
            intra = max(intra, np.max(cdist(clusters[i], clusters[i])))

        for j in range(i + 1, len(clusters)):
            inter = min(inter, np.min(cdist(clusters[i], clusters[j])))

    return inter / intra if intra != 0 else 0


def xie_beni_index(X, labels, centers):
    n = len(X)
    dist = cdist(X, centers)
    min_dist = np.min(dist, axis=1)

    numerator = np.sum(min_dist ** 2)

    center_dist = cdist(centers, centers)
    np.fill_diagonal(center_dist, np.inf)

    denominator = n * (np.min(center_dist) ** 2)
    return numerator / denominator


# -----------------------------
# GENERAL TREND + KMEANS
# -----------------------------
class Trend_KMeans:
    def __init__(self, n_clusters=3):
        self.k = n_clusters

    def fit(self, X):
        print("Extracting general trends...")
        trend_weights = compute_general_trend(X)

        # Apply weights
        X_weighted = X * trend_weights

        print("Running KMeans...")
        kmeans = KMeans(
            n_clusters=self.k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42
        )

        labels = kmeans.fit_predict(X_weighted)

        self.labels = labels
        self.centers = kmeans.cluster_centers_

        return labels, self.centers


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    file_path = "/Users/souviksarkarjr./Downloads/IFGO/Dataset/Wave.csv"

    print("Loading data...")
    X = load_and_preprocess(file_path)

    model = Trend_KMeans(n_clusters=3)
    labels, centers = model.fit(X)

    print("\nCalculating metrics...")

    sil = silhouette_score(X, labels)
    db = davies_bouldin_index(X, labels, centers)
    dunn = dunn_index(X, labels)
    xb = xie_beni_index(X, labels, centers)
    ch = calinski_harabasz_score(X, labels)

    print("\nFinal Results (Trend + KMeans):")
    print(f"Silhouette Score      : {sil:.4f}")
    print(f"Davies-Bouldin Index : {db:.4f}")
    print(f"Dunn Index           : {dunn:.4f}")
    print(f"Xie-Beni Index       : {xb:.4f}")
    print(f"Calinski-Harabasz    : {ch:.4f}")