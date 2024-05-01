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


class DESParametersGenerator(ParametersGenerator):
    def __init__(
        self,
        lag: int,
        linspace_info: LinspaceInfo,
        parameters_generation_method: ParametersGenerationMethod,
        parameters_required: list[ParameterType],
        init_values_coeff: float = 0.5,
        long_term_coeff_range: tuple[float, float] = (0.0, 0.3),
        trend_coeff_range: tuple[float, float] = (0.0, 0.05),
    ):
        super().__init__(
            lag=lag,
            linspace_info=linspace_info,
            parameters_generation_method=parameters_generation_method,
            parameters_required=parameters_required,
        )
        self.init_values_coeff = init_values_coeff
        self.long_term_coeff_range = long_term_coeff_range
        self.trend_coeff_range = trend_coeff_range

    def generate_parameters(self, source_data: NDArray | None = None) -> NDArrayFloat64:
        return self.parameters_generation_method.generate_all_parameters(
            parameters_required=self.parameters_required,
            source_data=source_data,
        )

    def generate_init_values(
        self, source_data: NDArray | None = None
    ) -> NDArrayFloat64:
        return np.array(
            [
                self.parameters_generation_method.get_mean_value(source_data)
                * self.init_values_coeff,
                0.0,
            ]
        )


class DoubleExponentialSmoothing(Process):
    def __init__(
        self,
        linspace_info: LinspaceInfo,
        parameters_generation_method: ParametersGenerationMethod,
        init_values_coeff: float = 0.5,
        long_term_coeff_range: tuple[float, float] = (0.0, 0.3),
        trend_coeff_range: tuple[float, float] = (0.0, 0.05),
    ):
        self.long_term_coeff_range = long_term_coeff_range
        self.trend_coeff_range = trend_coeff_range
        super().__init__(
            linspace_info=linspace_info,
            parameters_generation_method=parameters_generation_method,
        )
        self._parameters_generator = DESParametersGenerator(
            lag=self.lag,
            linspace_info=self.linspace_info,
            parameters_generation_method=parameters_generation_method,
            parameters_required=self.parameters,
            init_values_coeff=init_values_coeff,
            long_term_coeff_range=long_term_coeff_range,
            trend_coeff_range=trend_coeff_range,
        )

    @property
    def name(self) -> str:
        return "double_exponential_smoothing"

    @property
    def parameters(self) -> list[ParameterType]:
        return [
            StdType(),
            CoefficientType(constraints=np.array(self.long_term_coeff_range)),
            CoefficientType(constraints=np.array(self.trend_coeff_range)),
        ]

    @property
    def lag(self) -> int:
        return 1

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
        ets_values.set_normal_error(mean=0.0, std=data[1][2])
        if previous_values is None:
            (
                long_term_init_value,
                trend_init_value,
            ) = self.parameters_generator.generate_init_values(source_data=source_data)
        else:
            long_term_init_value = previous_values[-1]
            trend_init_value = self.parameters_generator.generate_init_values(
                source_data=source_data
            )[1]
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
    proc = DoubleExponentialSmoothing(test_generator_linspace, method)
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
