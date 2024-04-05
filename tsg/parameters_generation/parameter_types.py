import math

import numpy as np
from numpy.typing import NDArray
from math import log10

from tsg.linspace_info import LinspaceInfo


def increase_dimension(values: NDArray[np.float64], params_num: int):
    values_num = len(values)
    new_values = np.array([0. for _ in range(params_num)])


def convert_to_range(values: NDArray[np.float64], ranges: NDArray[np.float64]) -> NDArray[np.float64]:
    converted_values = values.copy()
    min_value = np.min(values)
    max_value = np.max(values)
    low_border = 10 ** (int(log10(min_value)))
    high_border = 10 ** (int(log10(max_value)) + 1)
    for i in range(len(values)):
        value = values[i]
        start = ranges[i][0]
        end = ranges[i][1]
        converted_value_coeff = (value - low_border) / (high_border - low_border)
        converted_value = (end - start) * converted_value_coeff + start
        converted_values[i] = converted_value
    return converted_values
