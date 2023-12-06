from numpy import array, average, flip, max, min
from numpy.typing import NDArray


class AggregatedData:
    def __init__(
        self, source_data: NDArray, weighted_values: bool = True, use_max: bool = True
    ):
        self.source_data = source_data
        self.num_values = source_data.shape[0]
        self.sum_values = source_data.sum()
        self.max_value = max(source_data)
        self.min_value = min(source_data)
        self.weights = self.calculate_weights() if weighted_values else None
        self.mean_value = average(source_data, weights=self.weights)
        self.fraction = (
            self.mean_value / self.max_value
            if use_max
            else self.mean_value / self.sum_values
        )

    def calculate_weights(self) -> NDArray:
        progression_sum = (1 + self.num_values) * self.num_values / 2
        values = array([i + 1 for i in range(self.num_values)])
        return flip(values) / progression_sum
