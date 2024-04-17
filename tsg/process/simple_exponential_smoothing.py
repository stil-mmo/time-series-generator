import os
from typing import Tuple

import hydra
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.parameters_generation.parameter_types import (
    CoefficientType,
    ParameterType,
    StdType,
)
from tsg.parameters_generation.parameters_generation_method import (
    ParametersGenerationMethod,
)
from tsg.parameters_generation.random_method import RandomMethod
from tsg.process.ets_process_resources.ets_process_builder import ETSProcessBuilder
from tsg.process.process import ParametersGenerator, Process
from tsg.time_series import TimeSeries
from tsg.utils.utils import draw_process_plot


class SESParametersGenerator(ParametersGenerator):
    def __init__(
        self,
        lag: int,
        linspace_info: LinspaceInfo,
        parameters_generation_method: ParametersGenerationMethod,
        parameters_required: list[ParameterType],
        init_values_coeff: float = 0.5,
        long_term_coeff_range: Tuple[float, float] = (0.0, 0.3),
    ):
        super().__init__(
            lag=lag,
            linspace_info=linspace_info,
            parameters_generation_method=parameters_generation_method,
            parameters_required=parameters_required,
        )
        self.init_values_coeff = init_values_coeff
        self.long_term_coeff_range = long_term_coeff_range

    def generate_parameters(self) -> NDArray[np.float64]:
        return self.parameters_generation_method.generate_all_parameters(
            parameters_required=self.parameters_required
        )

    def generate_init_values(self) -> NDArray[np.float64]:
        return np.array(
            [self.parameters_generation_method.mean_value * self.init_values_coeff]
        )


class SimpleExponentialSmoothing(Process):
    def __init__(
        self,
        linspace_info: LinspaceInfo,
        parameters_generation_method: ParametersGenerationMethod,
        init_values_coeff: float = 0.5,
        long_term_coeff_range: Tuple[float, float] = (0.0, 0.3),
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


@hydra.main(
    version_base="1.2", config_path=os.path.join("..", ".."), config_name="config"
)
def show_plot(cfg: DictConfig):
    test_generator_linspace = LinspaceInfo(np.float64(0.0), np.float64(100.0), 100)
    method = RandomMethod(cfg.parameters_generation_method, test_generator_linspace)
    proc = SimpleExponentialSmoothing(test_generator_linspace, method)
    test_sample = (100, proc.parameters_generator.generate_parameters())
    ts, info = proc.generate_time_series(test_sample)
    draw_process_plot(ts, info)


if __name__ == "__main__":
    show_plot()
