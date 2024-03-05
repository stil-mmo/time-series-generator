import numpy as np
from numpy.typing import NDArray

from tsg.linspace_info import LinspaceInfo
from tsg.process.ets_process_resources.ets_process_builder import ETSProcessBuilder
from tsg.process.process import ParametersGenerator, Process
from tsg.sampling.aggregated_data import AggregatedData
from tsg.time_series import TimeSeries
from tsg.utils.utils import draw_process_plot


class DESParametersGenerator(ParametersGenerator):
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

    def generate_parameters(self) -> NDArray[np.float64]:
        if self.aggregated_data is None:
            std = self.linspace_info.generate_std()
            long_term_coefficient = np.random.uniform(0.0, 0.3)
            trend_coefficient = np.random.uniform(0.0, 0.05)
        else:
            std = self.linspace_info.generate_std(
                source_value=self.aggregated_data.fraction
            )
            long_term_coefficient = self.aggregated_data.fraction / 3
            trend_coefficient = long_term_coefficient / 20.0
        return np.array([long_term_coefficient, trend_coefficient, std])

    def generate_init_values(self) -> NDArray[np.float64]:
        if self.aggregated_data is None:
            init_values = self.linspace_info.generate_values(
                num_values=2, is_normal=False
            )
            init_values[1] = 0.0
        else:
            init_values = np.array([self.aggregated_data.mean_value / 2, 0.0])
        return init_values


class DoubleExponentialSmoothing(Process):
    def __init__(
        self,
        linspace_info: LinspaceInfo,
        lag: int = 1,
        aggregated_data: AggregatedData | None = None,
    ):
        parameters_generator = DESParametersGenerator(
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
        return "double_exponential_smoothing"

    @property
    def num_parameters(self) -> int:
        return 3

    def generate_time_series(
        self,
        data: tuple[int, NDArray[np.float64]],
        previous_values: NDArray[np.float64] | None = None,
    ) -> tuple[TimeSeries, dict]:
        ets_values = ETSProcessBuilder(data[0])
        ets_values.set_normal_error(mean=0.0, std=data[1][2])
        if previous_values is None:
            (
                long_term_init_value,
                trend_init_value,
            ) = self.parameters_generator.generate_init_values()
        else:
            long_term_init_value = previous_values[-1]
            trend_init_value = self.parameters_generator.generate_init_values()[1]
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
            data, np.array([long_term_init_value, trend_init_value])
        )


if __name__ == "__main__":
    test_generator_linspace = LinspaceInfo(np.float64(0.0), np.float64(100.0), 100)
    proc = DoubleExponentialSmoothing(test_generator_linspace)
    test_sample = (100, proc.parameters_generator.generate_parameters())
    ts, info = proc.generate_time_series(test_sample)
    draw_process_plot(ts, info)
