import numpy as np
from numpy.typing import NDArray

from tsg.linspace_info import LinspaceInfo
from tsg.process.ets_process_resources.ets_process_builder import ETSProcessBuilder
from tsg.process.process import ParametersGenerator, Process
from tsg.sampling.aggregated_data import AggregatedData
from tsg.time_series import TimeSeries
from tsg.utils.utils import draw_process_plot


class SESParametersGenerator(ParametersGenerator):
    def __init__(
        self,
        lag: int,
        linspace_info: LinspaceInfo,
        aggregated_data: AggregatedData | None = None,
    ):
        super().__init__(
            lag=lag,
            linspace_info=linspace_info,
            aggregated_data=aggregated_data,
        )

    def generate_parameters(self) -> tuple[float, ...]:
        if self.aggregated_data is None:
            std = self.linspace_info.generate_std()
            error_coefficient = np.random.uniform(0.0, 0.3)
        else:
            std = self.linspace_info.generate_std(
                source_value=self.aggregated_data.fraction
            )
            error_coefficient = self.aggregated_data.fraction / 3
        return error_coefficient, std

    def generate_init_values(self) -> NDArray:
        if self.aggregated_data is None:
            values = self.linspace_info.generate_values(is_normal=False)
        else:
            values = np.array([self.aggregated_data.mean_value / 2])
        return values


class SimpleExponentialSmoothing(Process):
    def __init__(
        self,
        linspace_info: LinspaceInfo,
        lag: int = 1,
        aggregated_data: AggregatedData | None = None,
    ):
        parameters_generator = SESParametersGenerator(
            lag=lag,
            linspace_info=linspace_info,
            aggregated_data=aggregated_data,
        )
        super().__init__(
            lag=lag,
            linspace_info=linspace_info,
            parameters_generator=parameters_generator,
            aggregated_data=aggregated_data,
        )

    @property
    def name(self) -> str:
        return "simple_exponential_smoothing"

    @property
    def num_parameters(self) -> int:
        return 2

    def generate_time_series(
        self,
        data: tuple[int, tuple[float, ...]],
        previous_values: NDArray | None = None,
    ) -> tuple[TimeSeries, dict]:
        ets_values = ETSProcessBuilder(data[0])
        ets_values.set_normal_error(mean=0.0, std=data[1][1])
        if previous_values is None:
            init_value = self.parameters_generator.generate_init_values()[0]
        else:
            init_value = previous_values[-1]
        ets_values.set_long_term(init_value=init_value, parameter=data[1][0])
        exp_time_series = TimeSeries(data[0])
        exp_time_series.add_values(ets_values.generate_values(), (self.name, data))
        return exp_time_series, self.get_info(data, init_value)


if __name__ == "__main__":
    test_generator_linspace = LinspaceInfo(0.0, 100.0, 100)
    proc = SimpleExponentialSmoothing(test_generator_linspace)
    test_sample = (100, proc.parameters_generator.generate_parameters())
    ts, info = proc.generate_time_series(test_sample)
    draw_process_plot(ts, info)
