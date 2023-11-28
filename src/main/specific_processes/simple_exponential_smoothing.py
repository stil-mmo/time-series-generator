from numpy import array, sqrt
from numpy.random import uniform
from numpy.typing import NDArray

from src.main.generator_linspace import GeneratorLinspace
from src.main.process import Process
from src.main.specific_processes.ets_process_resources.ets_process_builder import (
    ETSProcessBuilder,
)
from src.main.time_series import TimeSeries
from src.main.utils.parameters_approximation import weighted_mean
from src.main.utils.utils import draw_process_plot


class SimpleExponentialSmoothing(Process):
    def __init__(self, generator_linspace: GeneratorLinspace, lag: int = 1):
        super().__init__(lag, generator_linspace)

    @property
    def name(self) -> str:
        return "simple_exponential_smoothing"

    @property
    def num_parameters(self) -> int:
        return 2

    def calculate_data(
        self, source_values: NDArray | None = None
    ) -> tuple[tuple[float, ...], NDArray]:
        if source_values is None:
            return self.generate_parameters(), self.generate_init_values()
        mean = weighted_mean(source_values)
        error_coefficient = mean / self.generator_linspace.stop
        std = self.generator_linspace.calculate_std(mean)
        return (error_coefficient, sqrt(std)), array([mean])

    def generate_parameters(self) -> tuple[float, ...]:
        std = self.generator_linspace.generate_std()
        error_coefficient = uniform(0.0, 1.0)
        return error_coefficient, std

    def generate_init_values(self) -> NDArray:
        return self.generator_linspace.generate_values(is_normal=False)

    def generate_time_series(
        self,
        sample: tuple[int, tuple[float, ...]],
        previous_values: NDArray | None = None,
    ) -> tuple[TimeSeries, dict]:
        ets_values = ETSProcessBuilder(sample[0])
        ets_values.set_normal_error(mean=0.0, std=sample[1][1])
        if previous_values is None:
            init_value = self.generate_init_values()[0]
        else:
            init_value = previous_values[-1]
        ets_values.set_long_term(init_value=init_value, parameter=sample[1][0])
        exp_time_series = TimeSeries(sample[0])
        exp_time_series.add_values(ets_values.generate_values(), (self.name, sample))
        return exp_time_series, self.get_info(sample, init_value)


if __name__ == "__main__":
    test_generator_linspace = GeneratorLinspace(0.0, 100.0, 100)
    proc = SimpleExponentialSmoothing(test_generator_linspace)
    test_sample = (100, proc.generate_parameters())
    ts, info = proc.generate_time_series(test_sample)
    draw_process_plot(ts, info)
