import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from tsg.sampling.point_sampling import get_border_value, move_points
from tsg.utils.typing import NDArrayFloat64T, NDArrayIntT


def cluster_points(points: NDArrayFloat64T, n_clusters: int) -> NDArrayIntT:
    model = KMeans(n_clusters=n_clusters)
    model.fit(points)
    yhat = model.predict(points)
    return np.array(yhat, dtype=np.int_)


def get_blobs(
    num_samples: int, centers: int
) -> tuple[NDArrayFloat64T, NDArrayFloat64T, tuple[float, float], float]:
    X, y = make_blobs(
        n_samples=num_samples, n_features=3, centers=centers, center_box=(-1.0, 1.0)
    )
    shift = move_points(X)
    border_values = (
        get_border_value(X, is_min=True),
        get_border_value(X, is_min=False),
    )
    return X, y, border_values, shift
