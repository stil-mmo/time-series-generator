from numpy import zeros, average, array, flip, cumsum, insert, std
from numpy.typing import NDArray
from numpy.linalg import norm


def calculate_distance(point: NDArray) -> float:
    zero = zeros(point.shape)
    return norm(point - zero)


def calculate_weights(num_values: int) -> NDArray:
    progression_sum = (1 + num_values) * num_values / 2
    values = array([i + 1 for i in range(num_values)])
    return flip(values) / progression_sum


def weighted_mean(values: NDArray, weights: NDArray | None = None) -> float:
    result_weights = calculate_weights(values.shape[0]) if weights is None else weights
    return average(values, weights=result_weights)


def weighted_std(values: NDArray, weights: NDArray | None = None) -> float:
    result_weights = calculate_weights(values.shape[0]) if weights is None else weights
    return (
        norm(values - weighted_mean(values, weights=result_weights)) / values.shape[0]
    )


def moving_average(values: NDArray, window_size: int) -> NDArray:
    cum_sum = cumsum(insert(values, 0, 0))
    return (cum_sum[window_size:] - cum_sum[:-window_size]) / float(window_size)


if __name__ == "__main__":
    print(moving_average(array([100, 15, 67]), 2))
    print(std(moving_average(array([100, 15, 67]), 2)))
    print(moving_average(array([98, 17, 70]), 2))
    print(std(moving_average(array([98, 17, 70]), 2)))
    print(moving_average(array([15, 67, 100]), 2))
    print(std(moving_average(array([15, 67, 100]), 2)))
