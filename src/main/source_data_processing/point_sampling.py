import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray


def sample_spherical(num_points: int, ndim=3) -> NDArray:
    vec = np.randn(ndim, num_points)
    vec /= np.norm(vec, axis=0)
    return vec


def get_border_value(coordinates: NDArray, is_min: bool = True) -> float:
    if is_min:
        return np.min(coordinates)
    else:
        return np.max(coordinates)


def move_points(coordinates: NDArray) -> NDArray:
    min_coordinate = get_border_value(coordinates)
    if min_coordinate < 0:
        coordinates += abs(min_coordinate)
    return coordinates


def sample_points(num_points: int) -> tuple[NDArray, tuple[float, float]]:
    coordinates = move_points(sample_spherical(num_points))
    border_values = (
        get_border_value(coordinates, is_min=True),
        get_border_value(coordinates, is_min=False),
    )
    return np.transpose(coordinates), border_values


def show_sphere(coordinates: NDArray) -> None:
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d", "aspect": "equal"})
    ax.plot_wireframe(x, y, z, color="k", rstride=1, cstride=1)
    ax.scatter(coordinates[0], coordinates[1], coordinates[2], s=100, c="r", zorder=10)


if __name__ == "__main__":
    show_sphere(sample_spherical(10))
