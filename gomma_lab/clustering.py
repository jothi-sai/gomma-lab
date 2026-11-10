
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def kmeans_clustering(k=3, samples=500, iterations=10):
    X, _ = make_blobs(
        n_samples=samples,
        centers=k,
        n_features=2,
        random_state=23
    )

    rng = np.random.default_rng(23)
    centers = X[rng.choice(len(X), k, replace=False)].copy()

    for i in range(iterations):
        distances = np.linalg.norm(
            X[:, None] - centers[None, :], axis=2
        )
        labels = np.argmin(distances, axis=1)

        for j in range(k):
            if np.any(labels == j):
                centers[j] = X[labels == j].mean(axis=0)

    distances = np.linalg.norm(
        X[:, None] - centers[None, :], axis=2
    )
    labels = np.argmin(distances, axis=1)

    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(
        centers[:, 0], centers[:, 1],
        marker="*", c="red", s=200
    )
    plt.title("K-Means Clustering")
    plt.show()

    return centers, labels
