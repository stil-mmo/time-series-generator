import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray


def sample_spherical(num_points: int, ndim=3) -> NDArray:
    vec = np.random.randn(ndim, num_points)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def get_border_value(coordinates: NDArray, is_min: bool = True) -> float:
    if is_min:
        return np.min(coordinates)
    else:
        return np.max(coordinates)


def move_points(coordinates: NDArray) -> float:
    min_coordinate = get_border_value(coordinates)
    if min_coordinate < 0:
        coordinates += abs(min_coordinate)
    return abs(min_coordinate)


def sample_points(num_points: int) -> tuple[NDArray, tuple[float, float], float]:
    coordinates = sample_spherical(num_points)
    shift = move_points(coordinates)
    border_values = (
        get_border_value(coordinates, is_min=True),
        get_border_value(coordinates, is_min=False),
    )
    return np.transpose(coordinates), border_values, shift


def show_sphere(shift: float, coordinates: NDArray) -> None:
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))
    x += shift
    y += shift
    z += shift

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d", "aspect": "equal"})
    ax.plot_wireframe(x, y, z, color="grey", alpha=0.3)
    ax.scatter(coordinates[0][0], coordinates[0][1], coordinates[0][2], s=100, c="blue")
    ax.scatter(
        coordinates[1][0], coordinates[1][1], coordinates[1][2], s=100, c="orange"
    )
    ax.scatter(
        coordinates[2][0], coordinates[2][1], coordinates[2][2], s=100, c="green"
    )
    ax.scatter(coordinates[3][0], coordinates[3][1], coordinates[3][2], s=100, c="red")
    ax.scatter(
        coordinates[4][0], coordinates[4][1], coordinates[4][2], s=100, c="black"
    )
