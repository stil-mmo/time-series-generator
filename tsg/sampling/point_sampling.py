import numpy as np

from tsg.utils.typing import NDArrayFloat64


def sample_points(
    num_points: int,
) -> tuple[NDArrayFloat64, tuple[float, float], float]:
    coordinates = sample_spherical(num_points)
    shift = move_points(coordinates)
    border_values = (
        get_border_value(coordinates, is_min=True),
        get_border_value(coordinates, is_min=False),
    )
    return np.transpose(coordinates), border_values, shift


def sample_spherical(num_points: int, ndim=3) -> NDArrayFloat64:
    vec = np.random.randn(ndim, num_points)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def get_border_value(coordinates: NDArrayFloat64, is_min: bool = True) -> float:
    if is_min:
        return float(np.min(coordinates))
    else:
        return float(np.max(coordinates))


def move_points(coordinates: NDArrayFloat64) -> float:
    min_coordinate = get_border_value(coordinates)
    shift = 0.0
    if min_coordinate < 0:
        coordinates += np.abs(min_coordinate) * 2
        shift = np.abs(min_coordinate) * 2
    return shift
