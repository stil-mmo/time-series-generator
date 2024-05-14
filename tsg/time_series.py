import numpy as np

from tsg.utils.typing import NDArrayFloat64T, ProcessConfigT


class TimeSeries:
    def __init__(self, num_steps: int) -> None:
        self.num_steps = num_steps
        self.last_index = 0
        self.values = np.zeros(shape=(1, num_steps))[0]
        self.metadata: list[ProcessConfigT] = []

    def add_values(
        self,
        new_values: NDArrayFloat64T,
        new_metadata: ProcessConfigT,
    ) -> None:
        if self.last_index + len(new_values) > self.num_steps:
            print(
                f"Number of values to add in time series "
                f"exceeds the number of steps: {len(new_values)} > {self.num_steps}"
            )
        self.values[self.last_index : self.last_index + len(new_values)] = new_values
        self.last_index += len(new_values)
        self.metadata.append(new_metadata)

    def get_values(
        self, start_index: int = 0, end_index: int | None = None
    ) -> NDArrayFloat64T:
        if end_index is None:
            return self.values[start_index : self.last_index]
        else:
            return self.values[start_index:end_index]
