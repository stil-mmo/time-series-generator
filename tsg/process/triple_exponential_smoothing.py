import numpy as np
from numpy.typing import NDArray

from tsg.linspace_info import LinspaceInfo
from tsg.parameters_generation.aggregation_method import AggregationMethod
from tsg.parameters_generation.parameter_types import (
    CoefficientType,
    ParameterType,
    StdType,
)
from tsg.parameters_generation.parameters_generation_method import (
    ParametersGenerationMethod,
)
from tsg.process.ets_process_resources.ets_process_builder import ETSProcessBuilder
from tsg.process.process import ParametersGenerator, Process
from tsg.time_series import TimeSeries
from tsg.utils.typing import NDArrayFloat64
from tsg.utils.utils import draw_process_plot


class TESParametersGenerator(ParametersGenerator):
    def __init__(
        self,
        lag: int,
        linspace_info: LinspaceInfo,
        parameters_generation_method: ParametersGenerationMethod,
        parameters_required: list[ParameterType],
        long_term_coeff_range: tuple[float, float] = (0.0, 0.3),
        trend_coeff_range: tuple[float, float] = (0.0, 0.05),
        seasonal_coeff_range: tuple[float, float] = (0.0, 0.05),
    ):
        super().__init__(
            lag=lag,
            linspace_info=linspace_info,
            parameters_generation_method=parameters_generation_method,
            parameters_required=parameters_required,
        )

        self.long_term_coeff_range = long_term_coeff_range
        self.trend_coeff_range = trend_coeff_range
        self.seasonal_coeff_range = seasonal_coeff_range

    def generate_parameters(self, source_data: NDArray | None = None) -> NDArrayFloat64:
        return self.parameters_generation_method.generate_all_parameters(
            parameters_required=self.parameters_required
        )

    def generate_init_values(
        self, source_data: NDArray | None = None
    ) -> NDArrayFloat64:
        init_values = np.zeros((3, self.lag))
        init_values[0][0] = self.parameters_generation_method.get_mean_value(
            source_data
        )
        for i in range(1, self.lag):
            init_values[2][i] = init_values[2][i - 1] + np.random.normal(
                0.0, self.linspace_info.step
            )
        return init_values


class TripleExponentialSmoothing(Process):
    def __init__(
        self,
        linspace_info: LinspaceInfo,
        parameters_generation_method: ParametersGenerationMethod,
        lag: int = 12,
        long_term_coeff_range: tuple[float, float] = (0.0, 0.3),
        trend_coeff_range: tuple[float, float] = (0.0, 0.05),
        seasonal_coeff_range: tuple[float, float] = (0.0, 0.05),
    ):
        self.long_term_coeff_range = long_term_coeff_range
        self.trend_coeff_range = trend_coeff_range
        self.seasonal_coeff_range = seasonal_coeff_range
        self._lag = lag
        super().__init__(
            linspace_info=linspace_info,
            parameters_generation_method=parameters_generation_method,
        )
        self._parameters_generator = TESParametersGenerator(
            lag=self.lag,
            linspace_info=self.linspace_info,
            parameters_generation_method=parameters_generation_method,
            parameters_required=self.parameters,
            long_term_coeff_range=long_term_coeff_range,
            trend_coeff_range=trend_coeff_range,
            seasonal_coeff_range=seasonal_coeff_range,
        )

    @property
    def name(self) -> str:
        return "triple_exponential_smoothing"

    @property
    def parameters(self) -> list[ParameterType]:
        return [
            StdType(),
            CoefficientType(constraints=np.array(self.long_term_coeff_range)),
            CoefficientType(constraints=np.array(self.trend_coeff_range)),
            CoefficientType(constraints=np.array(self.seasonal_coeff_range)),
        ]

    @property
    def lag(self) -> int:
        return self._lag

    @property
    def parameters_generator(self) -> ParametersGenerator:
        return self._parameters_generator

    def generate_time_series(
        self,
        data: tuple[int, NDArrayFloat64],
        previous_values: NDArrayFloat64 | None = None,
        source_data: NDArrayFloat64 | None = None,
    ) -> tuple[TimeSeries, dict]:
        ets_values = ETSProcessBuilder(data[0])
        ets_values.set_normal_error(mean=0.0, std=data[1][3])
        if previous_values is None or len(previous_values) < 1:
            init_values = self.parameters_generator.generate_init_values(
                source_data=source_data
            )
            long_term_init_value = init_values[0][0]
            trend_init_value = init_values[1][0]
            seasonality_init_values = init_values[2]
        elif len(previous_values) < self.lag:
            init_values = self.parameters_generator.generate_init_values(
                source_data=source_data
            )
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
    test_generator_linspace = LinspaceInfo(0.0, 100.0, 100)
    method = AggregationMethod(test_generator_linspace)
    proc = TripleExponentialSmoothing(test_generator_linspace, method)
    source_data = np.array([10.0, 50.0])
    test_sample = (
        100,
        proc.parameters_generator.generate_parameters(source_data=source_data),
    )
    test_time_series, test_info = proc.generate_time_series(
        test_sample, source_data=source_data
    )
    draw_process_plot(test_time_series, test_info)


if __name__ == "__main__":
    show_plot()
