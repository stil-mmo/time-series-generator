import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from src.main.source_data_processing.point_sampling import (
    get_border_value,
    move_points,
    sample_points,
)


def cluster_points(points: NDArray[np.float32], n_clusters: int) -> NDArray[np.float32]:
    model = KMeans(n_clusters=n_clusters)
    model.fit(points)
    yhat = model.predict(points)
    return yhat


def get_blobs(
    num_samples: int, centers: int
) -> tuple[NDArray[np.float32], NDArray[np.float32], tuple[float, float], float]:
    X, y = make_blobs(
        n_samples=num_samples, n_features=3, centers=centers, center_box=(-1.0, 1.0)
    )
    shift = move_points(X)
    border_values = (
        get_border_value(X, is_min=True),
        get_border_value(X, is_min=False),
    )
    return X, y, border_values, shift
