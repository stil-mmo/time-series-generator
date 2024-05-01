import numpy as np
from numpy.typing import NDArray

from tsg.linspace_info import LinspaceInfo
from tsg.parameters_generation.aggregation_method import AggregationMethod
from tsg.parameters_generation.parameter_types import ParameterType, StdType
from tsg.parameters_generation.parameters_generation_method import (
    ParametersGenerationMethod,
)
from tsg.process.process import ParametersGenerator, Process
from tsg.time_series import TimeSeries
from tsg.utils.typing import NDArrayFloat64T
from tsg.utils.utils import draw_process_plot


class RWParametersGenerator(ParametersGenerator):
    def __init__(
        self,
        lag: int,
        linspace_info: LinspaceInfo,
        parameters_generation_method: ParametersGenerationMethod,
        parameters_required: list[ParameterType],
        init_values_coeff: float = 0.5,
    ):
        super().__init__(
            lag=lag,
            linspace_info=linspace_info,
            parameters_generation_method=parameters_generation_method,
            parameters_required=parameters_required,
        )
        self.init_values_coeff = init_values_coeff

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


class RandomWalk(Process):
    def __init__(
        self,
        linspace_info: LinspaceInfo,
        parameters_generation_method: ParametersGenerationMethod,
        init_values_coeff: float = 0.5,
    ):
        super().__init__(
            linspace_info=linspace_info,
            parameters_generation_method=parameters_generation_method,
        )
        self._parameters_generator = RWParametersGenerator(
            lag=self.lag,
            linspace_info=self.linspace_info,
            parameters_generation_method=parameters_generation_method,
            parameters_required=self.parameters,
            init_values_coeff=init_values_coeff,
        )

    @property
    def name(self) -> str:
        return "random_walk"

    @property
    def parameters(self) -> list[ParameterType]:
        return [StdType()]

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
        values = np.array([0.0 for _ in range(0, data[0])])
        values_to_add = data[0]
        if previous_values is None:
            values[0] = self.parameters_generator.generate_init_values(
                source_data=source_data
            )[0]
            previous_value = values[0]
            values_to_add -= 1
        else:
            previous_value = previous_values[-1]
        for i in range(data[0] - values_to_add, data[0]):
            previous_value += np.random.normal(0.0, data[1][0])
            values[i] = previous_value
        rw_time_series = TimeSeries(data[0])
        rw_time_series.add_values(values, (self.name, data))
        if previous_values is None:
            return rw_time_series, self.get_info(
                data, np.array(values[0 : data[0] - values_to_add])
            )
        else:
            return rw_time_series, self.get_info(data, np.array([previous_values[-1]]))


def show_plot():
    test_generator_linspace = LinspaceInfo(0.0, 100.0, 100)
    method = AggregationMethod(test_generator_linspace)
    proc = RandomWalk(test_generator_linspace, method)
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
