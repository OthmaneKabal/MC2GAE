import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances







def kmeans_classify_with_centroid_flag(gs_embeddings, k):
    """
    Performs KMeans clustering and flags the term closest to each centroid.

    :param gs_embeddings: dict {term: embedding}
    :param k: number of clusters
    :return: DataFrame with columns [term, cluster_id, is_centroid]
    """
    # Préparer données
    terms = list(gs_embeddings.keys())
    X = np.array([gs_embeddings[t] for t in terms])
    X_norm = normalize(X)  # pour se rapprocher du comportement cosinus

    # Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_norm)

    # Base DataFrame
    df = pd.DataFrame({
        'term': terms,
        'cluster_id': labels,
        'is_centroid': False
    })

    # Identifier le terme le plus proche du centroïde pour chaque cluster
    for cluster_id in range(k):
        cluster_indices = df[df['cluster_id'] == cluster_id].index
        cluster_vectors = X_norm[cluster_indices]
        centroid = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
        # Calculer les distances euclidiennes
        distances = euclidean_distances(cluster_vectors, centroid).flatten()
        closest_idx = cluster_indices[np.argmin(distances)]

        df.at[closest_idx, 'is_centroid'] = True
    df.to_excel("clustering_with_centroids.xlsx", index=False)
    print("Clustering result is exported")
    return df



def kmeans_with_fixed_centroids(gs_embeddings, core_concepts_embeddings):
    """
    Performs KMeans clustering with manually initialized centroids (core concepts),
    using cosine-friendly normalization, and flags the most representative term of each cluster.

    :param gs_embeddings: dict {term: embedding}
    :param core_concepts_embeddings: dict {core_concept_name: embedding}
    :return: DataFrame with columns [term, cluster_id, is_centroid]
    """
    # Récupérer et normaliser les données
    terms = list(gs_embeddings.keys())
    X = np.array([gs_embeddings[term] for term in terms])
    X_norm = normalize(X)

    # Initialisation des centroïdes depuis les core concepts (normalisés aussi)
    initial_centroids = normalize(np.array([
        core_concepts_embeddings[c] for c in core_concepts_embeddings
    ]))

    k = len(core_concepts_embeddings)

    # KMeans avec initialisation forcée
    kmeans = KMeans(
        n_clusters=k,
        init=initial_centroids,
        n_init=1,
        random_state=42
    )

    labels = kmeans.fit_predict(X_norm)

    # Construction du DataFrame de résultat
    df = pd.DataFrame({
        'term': terms,
        'cluster_id': labels,
        'is_centroid': False
    })

    # Identifier pour chaque cluster le terme le plus proche de son centroïde
    for cluster_id in range(k):
        cluster_indices = df[df['cluster_id'] == cluster_id].index
        cluster_vectors = X_norm[cluster_indices]
        centroid = kmeans.cluster_centers_[cluster_id].reshape(1, -1)

        distances = euclidean_distances(cluster_vectors, centroid).flatten()
        closest_idx = cluster_indices[np.argmin(distances)]

        df.at[closest_idx, 'is_centroid'] = True
    df.to_excel("clustering_with_centroids_init.xlsx", index=False)
    print("Clustering result is exported")
    return df

















from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

def dbscan_classify_with_centroid_flag_cosine(gs_embeddings, eps=0.4, min_samples=3):
    """
    DBSCAN clustering (cosine distance) and flag centroid term (closest to cluster mean).

    :param gs_embeddings: dict {term: embedding}
    :param eps: DBSCAN epsilon (cosine distance threshold, e.g. 0.3)
    :param min_samples: minimum points to form a cluster
    :return: DataFrame [term, cluster_id, is_centroid]
    """
    terms = list(gs_embeddings.keys())
    X = np.array([gs_embeddings[t] for t in terms])

    # DBSCAN with cosine distance
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = db.fit_predict(X)

    df = pd.DataFrame({
        'term': terms,
        'cluster_id': labels,
        'is_centroid': False
    })

    valid_cluster_ids = [cid for cid in set(labels) if cid != -1]

    for cluster_id in valid_cluster_ids:
        cluster_indices = df[df['cluster_id'] == cluster_id].index
        cluster_vectors = X[cluster_indices]
        centroid = np.mean(cluster_vectors, axis=0).reshape(1, -1)

        distances = cosine_distances(cluster_vectors, centroid).flatten()
        closest_idx = cluster_indices[np.argmin(distances)]

        df.at[closest_idx, 'is_centroid'] = True
    df.to_excel("clustering_with_centroids_DBSCAN.xlsx", index=False)
    print("Clustering result is exported")
    return df