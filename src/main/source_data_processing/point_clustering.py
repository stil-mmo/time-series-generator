from numpy import float32
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from src.main.source_data_processing.point_sampling import sample_points


def cluster_points(points: NDArray[float32], n_clusters: int) -> NDArray[float32]:
    model = KMeans(n_clusters=n_clusters)
    model.fit(points)
    yhat = model.predict(points)
    return yhat


if __name__ == "__main__":
    sample_points, _ = sample_points(4)
    print(sample_points)
    cluster_points(points=sample_points, n_clusters=2)
