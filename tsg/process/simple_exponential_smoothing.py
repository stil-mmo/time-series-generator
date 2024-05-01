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
from tsg.utils.typing import NDArrayFloat64T
from tsg.utils.utils import draw_process_plot


class SESParametersGenerator(ParametersGenerator):
    def __init__(
        self,
        lag: int,
        linspace_info: LinspaceInfo,
        parameters_generation_method: ParametersGenerationMethod,
        parameters_required: list[ParameterType],
        init_values_coeff: float = 0.5,
        long_term_coeff_range: tuple[float, float] = (0.0, 0.3),
    ):
        super().__init__(
            lag=lag,
            linspace_info=linspace_info,
            parameters_generation_method=parameters_generation_method,
            parameters_required=parameters_required,
        )
        self.init_values_coeff = init_values_coeff
        self.long_term_coeff_range = long_term_coeff_range

    def generate_parameters(
        self, source_data: NDArray | None = None
    ) -> NDArrayFloat64T:
        return self.parameters_generation_method.generate_all_parameters(
            parameters_required=self.parameters_required,
            source_data=source_data,
        )

    def generate_init_values(
        self, source_data: NDArray | None = None
    ) -> NDArrayFloat64T:
        return np.array(
            [
                self.parameters_generation_method.get_mean_value(source_data)
                * self.init_values_coeff
            ]
        )


class SimpleExponentialSmoothing(Process):
    def __init__(
        self,
        linspace_info: LinspaceInfo,
        parameters_generation_method: ParametersGenerationMethod,
        init_values_coeff: float = 0.5,
        long_term_coeff_range: tuple[float, float] = (0.0, 0.3),
    ):
        self.long_term_coeff_range = long_term_coeff_range
        super().__init__(
            linspace_info=linspace_info,
            parameters_generation_method=parameters_generation_method,
        )
        self._parameters_generator = SESParametersGenerator(
            lag=self.lag,
            linspace_info=self.linspace_info,
            parameters_generation_method=parameters_generation_method,
            parameters_required=self.parameters,
            init_values_coeff=init_values_coeff,
            long_term_coeff_range=long_term_coeff_range,
        )

    @property
    def name(self) -> str:
        return "simple_exponential_smoothing"

    @property
    def parameters(self) -> list[ParameterType]:
        return [
            StdType(),
            CoefficientType(constraints=np.array(self.long_term_coeff_range)),
        ]

    @property
    def lag(self) -> int:
        return 1

    @property
    def parameters_generator(self) -> ParametersGenerator:
        return self._parameters_generator

    def generate_time_series(
        self,
        data: tuple[int, NDArrayFloat64T],
        previous_values: NDArrayFloat64T | None = None,
        source_data: NDArrayFloat64T | None = None,
    ) -> tuple[TimeSeries, dict]:
        ets_values = ETSProcessBuilder(data[0])
        ets_values.set_normal_error(mean=0.0, std=data[1][1])
        if previous_values is None:
            init_value = self.parameters_generator.generate_init_values(
                source_data=source_data
            )[0]
        else:
            init_value = previous_values[-1]
        ets_values.set_long_term(init_value=init_value, parameter=data[1][0])
        exp_time_series = TimeSeries(data[0])
        exp_time_series.add_values(ets_values.generate_values(), (self.name, data))
        return exp_time_series, self.get_info(data, init_value)


def show_plot():
    test_generator_linspace = LinspaceInfo(0.0, 100.0, 100)
    method = AggregationMethod(test_generator_linspace)
    proc = SimpleExponentialSmoothing(test_generator_linspace, method)
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
