from numpy import array
from numpy.random import uniform
from numpy.typing import NDArray
from src.main.generator_linspace import GeneratorLinspace
from src.main.process.ets_process_resources.ets_process_builder import ETSProcessBuilder
from src.main.process.process import Process
from src.main.source_data_processing.aggregated_data import AggregatedData
from src.main.time_series import TimeSeries
from src.main.utils.utils import draw_process_plot


class SimpleExponentialSmoothing(Process):
    def __init__(
        self,
        generator_linspace: GeneratorLinspace,
        lag: int = 1,
        aggregated_data: AggregatedData | None = None,
    ):
        super().__init__(lag, generator_linspace, aggregated_data)

    @property
    def name(self) -> str:
        return "simple_exponential_smoothing"

    @property
    def num_parameters(self) -> int:
        return 2

    def generate_parameters(self) -> tuple[float, ...]:
        if self.aggregated_data is None:
            std = self.generator_linspace.generate_std()
            error_coefficient = uniform(0.0, 1.0)
        else:
            std = self.generator_linspace.generate_std(
                source_value=self.aggregated_data.fraction
            )
            error_coefficient = self.aggregated_data.fraction
        return error_coefficient, std

    def generate_init_values(self) -> NDArray:
        if self.aggregated_data is None:
            values = self.generator_linspace.generate_values(is_normal=False)
        else:
            values = array([self.aggregated_data.mean_value])
        return values

    def generate_time_series(
        self,
        data: tuple[int, tuple[float, ...]],
        previous_values: NDArray | None = None,
    ) -> tuple[TimeSeries, dict]:
        ets_values = ETSProcessBuilder(data[0])
        ets_values.set_normal_error(mean=0.0, std=data[1][1])
        if previous_values is None:
            init_value = self.generate_init_values()[0]
        else:
            init_value = previous_values[-1]
        ets_values.set_long_term(init_value=init_value, parameter=data[1][0])
        exp_time_series = TimeSeries(data[0])
        exp_time_series.add_values(ets_values.generate_values(), (self.name, data))
        return exp_time_series, self.get_info(data, init_value)


if __name__ == "__main__":
    test_generator_linspace = GeneratorLinspace(0.0, 100.0, 100)
    proc = SimpleExponentialSmoothing(test_generator_linspace)
    test_sample = (100, proc.generate_parameters())
    ts, info = proc.generate_time_series(test_sample)
    draw_process_plot(ts, info)
