from numpy import array, zeros
from numpy.random import uniform
from numpy.typing import NDArray
from src.main.generator_linspace import GeneratorLinspace
from src.main.process import Process
from src.main.specific_processes.ets_process_resources.ets_process_builder import (
    ETSProcessBuilder,
)
from src.main.time_series import TimeSeries
from src.main.utils.utils import draw_process_plot


class TripleExponentialSmoothing(Process):
    def __init__(self, generator_linspace: GeneratorLinspace, lag: int):
        super().__init__(lag, generator_linspace)

    @property
    def name(self) -> str:
        return "triple_exponential_smoothing"

    @property
    def num_parameters(self) -> int:
        return 4

    def generate_parameters(self) -> tuple[float, ...]:
        std = abs(self.generate_value()) / self.generator_linspace.parts
        long_term_coefficient = uniform(0.0, 1.0)
        trend_coefficient = uniform(0.0, 0.1)
        seasonality_coefficient = uniform(0.0, 0.1)
        return long_term_coefficient, trend_coefficient, seasonality_coefficient, std

    def generate_init_values(self) -> NDArray:
        init_values = zeros((3, self.lag))
        init_values[0][0] = self.generate_value()
        init_values[1][0] = 0.0
        center = (self.generator_linspace.start + self.generator_linspace.stop) / 2
        init_values[2] = array(
            [uniform(0.5 * center, 1.5 * center) for _ in range(self.lag)]
        )
        return init_values

    def generate_time_series(
        self,
        sample: tuple[int, tuple[float, ...]],
        previous_values: NDArray | None = None,
    ) -> tuple[TimeSeries, dict]:
        ets_values = ETSProcessBuilder(sample[0])
        ets_values.set_normal_error(mean=0.0, std=sample[1][3])
        if previous_values is None or len(previous_values) < 1:
            init_values = self.generate_init_values()
            long_term_init_value = init_values[0][0]
            trend_init_value = init_values[1][0]
            seasonality_init_values = init_values[2]
        elif len(previous_values) < self.lag:
            init_values = self.generate_init_values()
            init_values[2][: len(previous_values)] = previous_values
            long_term_init_value = previous_values[-1]
            trend_init_value = init_values[1][0]
            seasonality_init_values = init_values[2]
        else:
            long_term_init_value = previous_values[-1]
            trend_init_value = 0.0
            seasonality_init_values = previous_values[-self.lag :]
        ets_values.set_seasonal(
            lag=self.lag, init_values=seasonality_init_values, parameter=sample[1][2]
        )
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
    proc = TripleExponentialSmoothing(test_generator_linspace, 12)
    test_sample = (100, proc.generate_parameters())
    test_time_series, test_info = proc.generate_time_series(test_sample)
    draw_process_plot(test_time_series, test_info)
