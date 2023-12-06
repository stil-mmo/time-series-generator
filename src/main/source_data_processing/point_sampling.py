from matplotlib import pyplot as plt
from numpy import cos, linspace, max, min, ones_like, outer, pi, sin, transpose
from numpy.linalg import norm
from numpy.random import randn
from numpy.typing import NDArray


def sample_spherical(num_points: int, ndim=3) -> NDArray:
    vec = randn(ndim, num_points)
    vec /= norm(vec, axis=0)
    return vec


def get_border_value(coordinates: NDArray, is_min: bool = True) -> float:
    if is_min:
        return min(coordinates)
    else:
        return max(coordinates)


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
    return transpose(coordinates), border_values


def show_sphere(coordinates: NDArray) -> None:
    phi = linspace(0, pi, 20)
    theta = linspace(0, 2 * pi, 40)
    x = outer(sin(theta), cos(phi))
    y = outer(sin(theta), sin(phi))
    z = outer(cos(theta), ones_like(phi))

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d", "aspect": "equal"})
    ax.plot_wireframe(x, y, z, color="k", rstride=1, cstride=1)
    ax.scatter(coordinates[0], coordinates[1], coordinates[2], s=100, c="r", zorder=10)


if __name__ == "__main__":
    show_sphere(sample_spherical(10))
