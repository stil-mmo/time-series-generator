from numpy import array
from numpy.typing import NDArray


class TimeSeries:
    def __init__(self, num_steps: int):
        self.num_steps = num_steps
        self.last_index = 0
        self.values = array([0.0 for _ in range(0, num_steps)])
        self.samples = []

    def add_values(
        self, new_values: NDArray, new_sample: tuple[str, tuple[int, tuple[float, ...]]]
    ) -> None:
        if self.last_index + len(new_values) > self.num_steps:
            print(f"Number of values to add in time series exceeds the number of steps: {len(new_values)} > {self.num_steps}")
        self.values[self.last_index : self.last_index + len(new_values)] = new_values
        self.last_index += len(new_values)
        self.samples.append(new_sample)

    def get_values(self, start_index: int = 0, end_index: int | None = None) -> NDArray:
        if end_index is None:
            return self.values[start_index:self.last_index]
        else:
            return self.values[start_index:end_index]
