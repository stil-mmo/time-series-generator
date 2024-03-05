import numpy as np
from numpy.typing import NDArray


def sample_points(
    num_points: int,
) -> tuple[NDArray[np.float64], tuple[np.float64, np.float64], np.float64]:
    coordinates = sample_spherical(num_points)
    shift = move_points(coordinates)
    border_values = (
        get_border_value(coordinates, is_min=True),
        get_border_value(coordinates, is_min=False),
    )
    return np.transpose(coordinates), border_values, shift


def sample_spherical(num_points: int, ndim=3) -> NDArray[np.float64]:
    vec = np.random.randn(ndim, num_points)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def get_border_value(
    coordinates: NDArray[np.float64], is_min: bool = True
) -> np.float64:
    if is_min:
        return np.float64(np.min(coordinates))
    else:
        return np.float64(np.max(coordinates))


def move_points(coordinates: NDArray[np.float64]) -> np.float64:
    min_coordinate = get_border_value(coordinates)
    shift = np.float64(0)
    if min_coordinate < 0:
        coordinates += abs(min_coordinate)
        shift = np.abs(min_coordinate)
    return shift
