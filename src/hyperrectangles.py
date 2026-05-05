import numpy as np
import hdbscan
from utils import load_align_mat

# Source code taken directly from ANTONIONLP/ANTONIO
# Repository: https://github.com/ANTONIONLP/ANTONIO
def calculate_hyperrectangle(points):
    # Calculate the hyperrectangle around the points
    min_max_list = np.full((points.shape[1], 2), [np.inf, -np.inf])
    for point in points:
        for i in range(len(point)):
            if point[i] < min_max_list[i][0]:
                min_max_list[i][0] = point[i]
            if point[i] > min_max_list[i][1]:
                min_max_list[i][1] = point[i]
    return min_max_list

def compute_hyperrectangles(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
) -> list[np.ndarray]:
    embeddings = embeddings.astype(np.float64)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="cosine", algorithm='generic')
    clusterer.fit(embeddings)

    labels = clusterer.labels_

    cluster_ids = sorted(set(labels) - {-1})

    if not cluster_ids:
        print("Warning: HDBSCAN found no clusters, falling back to a single cluster.")
        labels = np.zeros(len(embeddings), dtype=int)
        cluster_ids = [0]

    center_of_rect = np.stack([
        embeddings[labels == cid].mean(axis=0) for cid in cluster_ids
    ])

    for i, point in enumerate(embeddings):
        if labels[i] == -1:
            labels[i] = cluster_ids[np.argmin(np.linalg.norm(center_of_rect - point, axis=1))]


    rectangles = []
    align_matrices = []
    for cluster_id in cluster_ids:
        points = embeddings[labels == cluster_id]
        align_mat = load_align_mat("DATSET", "MODEL", points, False)
        points = points @ align_mat
        rect   = calculate_hyperrectangle(points)
        rectangles.append(rect)
        align_matrices.append(align_mat)

    return rectangles, align_matrices