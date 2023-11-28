from numpy import transpose
from point_sampling import sample_spherical, show_sphere
from sklearn.neighbors import NearestNeighbors


def get_knn(points, k):
    neighbours = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(points)
    knn_distances, knn_indices = neighbours.kneighbors(points)
    return knn_indices, knn_distances


if __name__ == "__main__":
    coordinates = sample_spherical(3)
    print(transpose(coordinates))
    show_sphere(coordinates)
    print(get_knn(transpose(coordinates), 2))
