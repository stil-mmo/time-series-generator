from typing import Tuple

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
        init_values_coeff: float = 0.5,
        long_term_coeff_range: Tuple[float, float] = (0.0, 0.3),
    ):
        super().__init__(
            lag=lag,
            linspace_info=linspace_info,
            aggregated_data=aggregated_data,
        )
        self.init_values_coeff = init_values_coeff
        self.long_term_coeff_range = long_term_coeff_range

    def generate_parameters(self) -> NDArray[np.float64]:
        if self.aggregated_data is None:
            std = self.linspace_info.generate_std()
            long_term_coefficient = np.random.uniform(
                self.long_term_coeff_range[0], self.long_term_coeff_range[1]
            )
        else:
            std = self.linspace_info.generate_std(
                source_value=self.aggregated_data.fraction
            )
            long_term_coefficient = (
                self.aggregated_data.fraction * self.long_term_coeff_range[1]
            )
        return np.array([long_term_coefficient, std])

    def generate_init_values(self) -> NDArray[np.float64]:
        if self.aggregated_data is None:
            values = self.linspace_info.generate_values(is_normal=False)
        else:
            values = np.array(
                [self.aggregated_data.mean_value * self.init_values_coeff]
            )
        return values


class SimpleExponentialSmoothing(Process):
    def __init__(
        self,
        linspace_info: LinspaceInfo,
        aggregated_data: AggregatedData | None = None,
        init_values_coeff: float = 0.5,
        long_term_coeff_range: Tuple[float, float] = (0.0, 0.3),
    ):
        super().__init__(
            linspace_info=linspace_info,
            aggregated_data=aggregated_data,
        )
        self._parameters_generator = SESParametersGenerator(
            lag=self.lag,
            linspace_info=self.linspace_info,
            aggregated_data=self.aggregated_data,
            init_values_coeff=init_values_coeff,
            long_term_coeff_range=long_term_coeff_range,
        )

    @property
    def name(self) -> str:
        return "simple_exponential_smoothing"

    @property
    def num_parameters(self) -> int:
        return 2

    @property
    def lag(self) -> int:
        return 1

    @property
    def parameters_generator(self) -> ParametersGenerator:
        return self._parameters_generator

    def generate_time_series(
        self,
        data: tuple[int, NDArray[np.float64]],
        previous_values: NDArray[np.float64] | None = None,
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


def show_plot():
    test_generator_linspace = LinspaceInfo(np.float64(0.0), np.float64(100.0), 100)
    proc = SimpleExponentialSmoothing(test_generator_linspace)
    test_sample = (100, proc.parameters_generator.generate_parameters())
    ts, info = proc.generate_time_series(test_sample)
    draw_process_plot(ts, info)


if __name__ == "__main__":
    show_plot()
