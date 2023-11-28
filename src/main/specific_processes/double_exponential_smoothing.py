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


class DoubleExponentialSmoothing(Process):
    def __init__(self, generator_linspace: GeneratorLinspace, lag: int = 1):
        super().__init__(lag, generator_linspace)

    @property
    def name(self) -> str:
        return "double_exponential_smoothing"

    @property
    def num_parameters(self) -> int:
        return 3

    def calculate_data(
        self, source_values: NDArray | None = None
    ) -> tuple[tuple[float, ...], NDArray]:
        if source_values is None:
            return self.generate_parameters(), self.generate_init_values()
        mean = weighted_mean(source_values)
        long_term_coefficient = mean / self.generator_linspace.stop
        trend_coefficient = long_term_coefficient / 10.0
        std = self.generator_linspace.calculate_std(mean)
        return (long_term_coefficient, trend_coefficient, sqrt(std)), array([mean])

    def generate_parameters(self) -> tuple[float, ...]:
        std = self.generator_linspace.generate_std()
        long_term_coefficient = uniform(0.0, 1.0)
        trend_coefficient = uniform(0.0, 0.05)
        return long_term_coefficient, trend_coefficient, std

    def generate_init_values(self) -> NDArray:
        init_values = self.generator_linspace.generate_values(
            num_values=2, is_normal=False
        )
        init_values[1] = 0.0
        return init_values

    def generate_time_series(
        self,
        sample: tuple[int, tuple[float, ...]],
        previous_values: NDArray | None = None,
    ) -> tuple[TimeSeries, dict]:
        ets_values = ETSProcessBuilder(sample[0])
        ets_values.set_normal_error(mean=0.0, std=sample[1][2])
        if previous_values is None:
            long_term_init_value, trend_init_value = self.generate_init_values()
        else:
            long_term_init_value = previous_values[-1]
            trend_init_value = self.generate_init_values()[1]
        trend_index = ets_values.set_trend(
            init_value=trend_init_value, parameter=sample[1][1]
        )
        ets_values.set_long_term(
            init_value=long_term_init_value,
            parameter=sample[1][0],
            add_component_indexes=[trend_index],
        )
        trend_time_series = TimeSeries(sample[0])
        trend_time_series.add_values(ets_values.generate_values(), (self.name, sample))
        return trend_time_series, self.get_info(
            sample, (long_term_init_value, trend_init_value)
        )


if __name__ == "__main__":
    test_generator_linspace = GeneratorLinspace(0.0, 100.0, 100)
    proc = DoubleExponentialSmoothing(test_generator_linspace)
    test_sample = (100, proc.generate_parameters())
    ts, info = proc.generate_time_series(test_sample)
    draw_process_plot(ts, info)
