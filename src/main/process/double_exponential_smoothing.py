from numpy import array
from numpy.random import uniform
from numpy.typing import NDArray
from src.main.generator_linspace import GeneratorLinspace
from src.main.process.ets_process_resources.ets_process_builder import ETSProcessBuilder
from src.main.process.process import Process
from src.main.source_data_processing.aggregated_data import AggregatedData
from src.main.time_series import TimeSeries
from src.main.utils.utils import draw_process_plot


class DoubleExponentialSmoothing(Process):
    def __init__(
        self,
        generator_linspace: GeneratorLinspace,
        lag: int = 1,
        aggregated_data: AggregatedData | None = None,
    ):
        super().__init__(lag, generator_linspace, aggregated_data)

    @property
    def name(self) -> str:
        return "double_exponential_smoothing"

    @property
    def num_parameters(self) -> int:
        return 3

    def generate_parameters(self) -> tuple[float, ...]:
        if self.aggregated_data is None:
            std = self.generator_linspace.generate_std()
            long_term_coefficient = uniform(0.0, 1.0)
            trend_coefficient = uniform(0.0, 0.05)
        else:
            std = self.generator_linspace.generate_std(
                source_value=self.aggregated_data.fraction
            )
            long_term_coefficient = self.aggregated_data.fraction
            trend_coefficient = long_term_coefficient / 20.0
        return long_term_coefficient, trend_coefficient, std

    def generate_init_values(self) -> NDArray:
        if self.aggregated_data is None:
            init_values = self.generator_linspace.generate_values(
                num_values=2, is_normal=False
            )
            init_values[1] = 0.0
        else:
            init_values = array([self.aggregated_data.mean_value, 0.0])
        return init_values

    def generate_time_series(
        self,
        data: tuple[int, tuple[float, ...]],
        previous_values: NDArray | None = None,
    ) -> tuple[TimeSeries, dict]:
        ets_values = ETSProcessBuilder(data[0])
        ets_values.set_normal_error(mean=0.0, std=data[1][2])
        if previous_values is None:
            long_term_init_value, trend_init_value = self.generate_init_values()
        else:
            long_term_init_value = previous_values[-1]
            trend_init_value = self.generate_init_values()[1]
        trend_index = ets_values.set_trend(
            init_value=trend_init_value, parameter=data[1][1]
        )
        ets_values.set_long_term(
            init_value=long_term_init_value,
            parameter=data[1][0],
            add_component_indexes=[trend_index],
        )
        trend_time_series = TimeSeries(data[0])
        trend_time_series.add_values(ets_values.generate_values(), (self.name, data))
        return trend_time_series, self.get_info(
            data, (long_term_init_value, trend_init_value)
        )


if __name__ == "__main__":
    test_generator_linspace = GeneratorLinspace(0.0, 100.0, 100)
    proc = DoubleExponentialSmoothing(test_generator_linspace)
    test_sample = (100, proc.generate_parameters())
    ts, info = proc.generate_time_series(test_sample)
    draw_process_plot(ts, info)
