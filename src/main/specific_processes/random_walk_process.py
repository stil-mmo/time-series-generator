from random import uniform

from numpy import array
from numpy.typing import NDArray

from src.main.process import Process
from src.main.time_series import TimeSeries


class RandomWalkProcess(Process):
    def __init__(
        self, border_values: tuple[float, float], lag: int = 1, walk: float = 1.0
    ):
        super().__init__(lag, border_values)
        self.walk = walk

    @property
    def name(self) -> str:
        return "random_walk"

    @property
    def num_parameters(self) -> int:
        return 2

    def generate_parameters(self) -> tuple[float, ...]:
        up_probability = uniform(0, 1)
        down_probability = 1 - up_probability
        return up_probability, down_probability

    def generate_init_values(self) -> NDArray:
        return array([uniform(self.border_values[0], self.border_values[1])])

    def generate_time_series(
        self,
        sample: tuple[int, tuple[float, ...]],
        previous_values: NDArray | None = None,
    ) -> tuple[TimeSeries, dict]:
        up_probability, down_probability = sample[1]
        values = array([0.0 for _ in range(0, sample[0])])
        values_to_add = sample[0]
        if previous_values is None:
            init_values = self.generate_init_values()
            values[0 : len(init_values)] = init_values
            values_to_add -= len(init_values)
            previous_value = init_values[-1]
        else:
            previous_value = previous_values[-1]
        for i in range(sample[0] - values_to_add, sample[0]):
            if uniform(0, 1) < up_probability:
                previous_value += self.walk
            else:
                previous_value -= self.walk
            values[i] = previous_value
        rw_time_series = TimeSeries(sample[0])
        rw_time_series.add_values(values, (self.name, sample))
        if previous_values is None:
            return rw_time_series, self.get_info(
                sample, values[0 : sample[0] - values_to_add]
            )
        else:
            return rw_time_series, self.get_info(sample, (previous_values[-1],))
