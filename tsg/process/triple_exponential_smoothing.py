from typing import Tuple

import hydra
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.process.ets_process_resources.ets_process_builder import ETSProcessBuilder
from tsg.process.process import ParametersGenerator, Process
from tsg.sampling.aggregated_data import AggregatedData
from tsg.time_series import TimeSeries
from tsg.utils.utils import draw_process_plot


class TESParametersGenerator(ParametersGenerator):
    def __init__(
        self,
        lag: int,
        linspace_info: LinspaceInfo,
        aggregated_data: AggregatedData | None = None,
        long_term_coeff_range: Tuple[float, float] = (0.0, 0.3),
        trend_coeff_range: Tuple[float, float] = (0.0, 0.05),
        seasonal_coeff_range: Tuple[float, float] = (0.0, 0.05),
    ):
        super().__init__(
            lag=lag,
            linspace_info=linspace_info,
            aggregated_data=aggregated_data,
        )

        self.long_term_coeff_range = long_term_coeff_range
        self.trend_coeff_range = trend_coeff_range
        self.seasonal_coeff_range = seasonal_coeff_range

    def generate_parameters(self) -> NDArray[np.float64]:
        if self.aggregated_data is None:
            std = self.linspace_info.generate_std()
            long_term_coefficient = np.random.uniform(
                self.long_term_coeff_range[0], self.long_term_coeff_range[1]
            )
            trend_coefficient = np.random.uniform(
                self.trend_coeff_range[0], self.trend_coeff_range[1]
            )
            seasonality_coefficient = np.random.uniform(
                self.seasonal_coeff_range[0], self.seasonal_coeff_range[1]
            )
        else:
            std = self.linspace_info.generate_std(
                source_value=self.aggregated_data.fraction
            )
            long_term_coefficient = (
                self.aggregated_data.fraction * self.long_term_coeff_range[1]
            )
            trend_coefficient = long_term_coefficient * self.trend_coeff_range[1]
            seasonality_coefficient = 1 - (
                self.aggregated_data.fraction * self.seasonal_coeff_range[0]
            )
        return np.array(
            [long_term_coefficient, trend_coefficient, seasonality_coefficient, std]
        )

    def generate_init_values(self) -> NDArray[np.float64]:
        if self.aggregated_data is None:
            init_values = np.zeros((3, self.lag))
            init_values[0][0] = self.linspace_info.generate_values()
            init_values[1][0] = 0.0
            init_values[2] = self.linspace_info.generate_values(num_values=self.lag)
        else:
            init_values = np.zeros((3, self.lag))
            init_values[0][0] = self.aggregated_data.mean_value
            init_values[1][0] = 0.0
            init_values[2][0] = 0.0
            for i in range(1, self.lag):
                init_values[2][i] = init_values[2][i - 1] + np.random.normal(
                    0.0, self.linspace_info.step
                )
        return init_values


class TripleExponentialSmoothing(Process):
    def __init__(
        self,
        linspace_info: LinspaceInfo,
        aggregated_data: AggregatedData | None = None,
        lag: int = 12,
        long_term_coeff_range: Tuple[float, float] = (0.0, 0.3),
        trend_coeff_range: Tuple[float, float] = (0.0, 0.05),
        seasonal_coeff_range: Tuple[float, float] = (0.0, 0.05),
    ):
        self._lag = lag
        super().__init__(
            linspace_info=linspace_info,
            aggregated_data=aggregated_data,
        )
        self._parameters_generator = TESParametersGenerator(
            lag=self.lag,
            linspace_info=self.linspace_info,
            aggregated_data=self.aggregated_data,
            long_term_coeff_range=long_term_coeff_range,
            trend_coeff_range=trend_coeff_range,
            seasonal_coeff_range=seasonal_coeff_range,
        )

    @property
    def name(self) -> str:
        return "triple_exponential_smoothing"

    @property
    def num_parameters(self) -> int:
        return 4

    @property
    def lag(self) -> int:
        return self._lag

    @property
    def parameters_generator(self) -> ParametersGenerator:
        return self._parameters_generator

    def generate_time_series(
        self,
        data: tuple[int, NDArray[np.float64]],
        previous_values: NDArray[np.float64] | None = None,
    ) -> tuple[TimeSeries, dict]:
        ets_values = ETSProcessBuilder(data[0])
        ets_values.set_normal_error(mean=0.0, std=data[1][3])
        if previous_values is None or len(previous_values) < 1:
            init_values = self.parameters_generator.generate_init_values()
            long_term_init_value = init_values[0][0]
            trend_init_value = init_values[1][0]
            seasonality_init_values = init_values[2]
        elif len(previous_values) < self.lag:
            init_values = self.parameters_generator.generate_init_values()
            long_term_init_value = previous_values[-1]
            trend_init_value = init_values[1][0]
            seasonality_init_values = np.array([0.0 for _ in range(self.lag)])
            seasonality_init_values[: len(previous_values)] = previous_values
            for i in range(len(previous_values), self.lag):
                seasonality_init_values[i] = np.random.normal(
                    previous_values[-1], self.linspace_info.step
                )
        else:
            long_term_init_value = previous_values[-1]
            trend_init_value = 0.0
            seasonality_init_values = previous_values[-self.lag :]
        ets_values.set_seasonal(
            lag=self.lag,
            init_values=seasonality_init_values / sum(seasonality_init_values),
            parameter=data[1][2],
        )
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


def show_plot():
    test_generator_linspace = LinspaceInfo(np.float64(0.0), np.float64(100.0), 100)
    proc = TripleExponentialSmoothing(test_generator_linspace)
    test_sample = (100, proc.parameters_generator.generate_parameters())
    test_time_series, test_info = proc.generate_time_series(test_sample)
    draw_process_plot(test_time_series, test_info)


if __name__ == "__main__":
    show_plot()
