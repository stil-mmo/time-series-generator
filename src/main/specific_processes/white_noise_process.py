from math import sqrt

from numpy import array
from numpy.random import normal, randint, uniform
from numpy.typing import NDArray
from src.main.generator_linspace import GeneratorLinspace
from src.main.process import Process
from src.main.time_series import TimeSeries


class WhiteNoiseProcess(Process):
    def __init__(self, generator_linspace: GeneratorLinspace, lag: int = 0):
        super().__init__(lag, generator_linspace)
        self.distributions = {0: normal, 1: uniform}

    @property
    def name(self) -> str:
        return "white_noise"

    @property
    def num_parameters(self) -> int:
        return 3

    def generate_parameters(self) -> tuple[float, ...]:
        distribution_id = randint(0, len(self.distributions.keys()))
        if distribution_id == 0:
            mean = self.generate_value()
            std = self.generate_value()
            return float(distribution_id), mean, sqrt(abs(std))
        else:
            low = self.generate_value()
            high = self.generate_value()
            return float(distribution_id), min(low, high), max(low, high)

    def generate_init_values(self) -> NDArray:
        return array([])

    def generate_time_series(
        self,
        sample: tuple[int, tuple[float, ...]],
        previous_values: NDArray | None = None,
    ) -> tuple[TimeSeries, dict]:
        distribution_id, *parameters = sample[1]
        wn_values = self.distributions[int(distribution_id)](
            size=sample[0], *parameters
        )
        wn_time_series = TimeSeries(sample[0])
        wn_time_series.add_values(wn_values, (self.name, sample))
        return wn_time_series, self.get_info(sample)
