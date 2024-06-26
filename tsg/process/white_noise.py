import numpy as np
from numpy.typing import NDArray

from tsg.linspace_info import LinspaceInfo
from tsg.parameters_generation.aggregation_method import AggregationMethod
from tsg.parameters_generation.parameter_types import MeanType, ParameterType, StdType
from tsg.parameters_generation.parameters_generation_method import (
    ParametersGenerationMethod,
)
from tsg.process.process import ParametersGenerator, Process
from tsg.time_series import TimeSeries
from tsg.utils.typing import NDArrayFloat64T
from tsg.utils.utils import draw_process_plot


class WNParametersGenerator(ParametersGenerator):
    def __init__(
        self,
        lag: int,
        linspace_info: LinspaceInfo,
        parameters_generation_method: ParametersGenerationMethod,
        parameters_required: list[ParameterType],
    ) -> None:
        super().__init__(
            lag=lag,
            linspace_info=linspace_info,
            parameters_generation_method=parameters_generation_method,
            parameters_required=parameters_required,
        )

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
        return np.array([])


class WhiteNoise(Process):
    def __init__(
        self,
        linspace_info: LinspaceInfo,
        parameters_generation_method: ParametersGenerationMethod,
    ) -> None:
        super().__init__(
            linspace_info=linspace_info,
            parameters_generation_method=parameters_generation_method,
        )
        self._parameters_generator = WNParametersGenerator(
            lag=self.lag,
            linspace_info=self.linspace_info,
            parameters_generation_method=parameters_generation_method,
            parameters_required=self.parameters,
        )

    @property
    def name(self) -> str:
        return "white_noise"

    @property
    def parameters(self) -> list[ParameterType]:
        return [MeanType(), StdType()]

    @property
    def lag(self) -> int:
        return 0

    @property
    def parameters_generator(self) -> ParametersGenerator:
        return self._parameters_generator

    def generate_time_series(
        self,
        data: tuple[int, NDArrayFloat64T],
        previous_values: NDArrayFloat64T | None = None,
        source_data: NDArrayFloat64T | None = None,
    ) -> tuple[TimeSeries, dict]:
        if previous_values is None:
            wn_values = np.random.normal(data[1][0], data[1][1], size=data[0])
        else:
            std = self.linspace_info.generate_std()
            wn_values = np.random.normal(previous_values[-1], std, data[0])
        wn_time_series = TimeSeries(data[0])
        wn_time_series.add_values(wn_values, (self.name, data))
        return wn_time_series, self.get_info(data)


if __name__ == "__main__":
    test_generator_linspace = LinspaceInfo(0.0, 100.0, 100)
    method = AggregationMethod(test_generator_linspace)
    white_noise_process = WhiteNoise(test_generator_linspace, method)
    time_series, info = white_noise_process.generate_time_series(
        (
            100,
            white_noise_process.parameters_generator.generate_parameters(
                source_data=np.array([10.0, 50.0])
            ),
        )
    )
    draw_process_plot(time_series, info)
