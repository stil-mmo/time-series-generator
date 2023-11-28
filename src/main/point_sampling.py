from matplotlib import pyplot as plt
from numpy import cos, linspace, max, min, ones_like, outer, pi, sin
from numpy.linalg import norm
from numpy.random import randn
from numpy.typing import NDArray


def sample_spherical(num_points: int, ndim=3) -> NDArray:
    vec = randn(ndim, num_points)
    vec /= norm(vec, axis=0)
    return vec


def get_border_value(coordinates: NDArray, is_min: bool = True) -> float:
    if is_min:
        min_x = min(coordinates[:, 0])
        min_y = min(coordinates[:, 1])
        min_z = min(coordinates[:, 2])
        return min([min_x, min_y, min_z])
    else:
        max_x = max(coordinates[:, 0])
        max_y = max(coordinates[:, 1])
        max_z = max(coordinates[:, 2])
        return max([max_x, max_y, max_z])


def move_points(coordinates: NDArray) -> NDArray:
    min_coordinate = get_border_value(coordinates)
    if min_coordinate < 0:
        coordinates += abs(min_coordinate) * ones_like(coordinates)
    return coordinates


def sample_points(num_points: int) -> tuple[NDArray, tuple[float, float]]:
    coordinates = move_points(sample_spherical(num_points))
    border_values = (
        get_border_value(coordinates, is_min=True),
        get_border_value(coordinates, is_min=False),
    )
    return coordinates, border_values


if __name__ == "__main__":
    phi = linspace(0, pi, 20)
    theta = linspace(0, 2 * pi, 40)
    x = outer(sin(theta), cos(phi))
    y = outer(sin(theta), sin(phi))
    z = outer(cos(theta), ones_like(phi))

    xi, yi, zi = sample_spherical(10)

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d", "aspect": "equal"})
    ax.plot_wireframe(x, y, z, color="k", rstride=1, cstride=1)
    ax.scatter(xi, yi, zi, s=100, c="r", zorder=10)
