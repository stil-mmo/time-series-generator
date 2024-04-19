import os.path

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
from tsg.process.process import ParametersGenerator, Process
from tsg.time_series import TimeSeries
from tsg.utils.utils import draw_process_plot


class SRWParametersGenerator(ParametersGenerator):
    def __init__(
        self,
        lag: int,
        linspace_info: LinspaceInfo,
        parameters_generation_method: ParametersGenerationMethod,
        parameters_required: list[ParameterType],
        init_values_coeff: float = 0.5,
        fixed_walk: float | None = None,
        fixed_up_probability: float | None = None,
    ):
        super().__init__(
            lag=lag,
            linspace_info=linspace_info,
            parameters_generation_method=parameters_generation_method,
            parameters_required=parameters_required,
        )
        self.init_values_coeff = init_values_coeff
        self.fixed_walk = fixed_walk
        self.fixed_up_probability = (
            fixed_up_probability
            if (fixed_up_probability is not None and 0 <= fixed_up_probability <= 1)
            else None
        )

    def generate_parameters(self) -> NDArray[np.float64]:
        parameters = self.parameters_generation_method.generate_all_parameters(
            parameters_required=self.parameters_required
        )
        if self.fixed_walk is not None:
            parameters[1] = self.fixed_walk
        return parameters

    def generate_init_values(self) -> NDArray[np.float64]:
        return np.array(
            [self.parameters_generation_method.mean_value * self.init_values_coeff]
        )


class SimpleRandomWalk(Process):
    def __init__(
        self,
        linspace_info: LinspaceInfo,
        parameters_generation_method: ParametersGenerationMethod,
        init_values_coeff: float = 0.5,
        fixed_walk: float | None = None,
        fixed_up_probability: float | None = None,
    ):
        super().__init__(
            linspace_info=linspace_info,
            parameters_generation_method=parameters_generation_method,
        )
        self._parameters_generator = SRWParametersGenerator(
            lag=self.lag,
            linspace_info=self.linspace_info,
            parameters_generation_method=parameters_generation_method,
            parameters_required=self.parameters,
            init_values_coeff=init_values_coeff,
            fixed_walk=fixed_walk,
            fixed_up_probability=fixed_up_probability,
        )

    @property
    def name(self) -> str:
        return "simple_random_walk"

    @property
    def parameters(self) -> list[ParameterType]:
        return [CoefficientType(np.array([0.0, 1.0])), StdType()]

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
        up_probability, walk = data[1]
        values = np.array([0.0 for _ in range(0, data[0])])
        values_to_add = data[0]
        if previous_values is None:
            init_values = self.parameters_generator.generate_init_values()
            values[0 : len(init_values)] = init_values
            values_to_add -= len(init_values)
            previous_value = init_values[-1]
        else:
            previous_value = previous_values[-1]
        for i in range(data[0] - values_to_add, data[0]):
            if np.random.uniform(0, 1) < up_probability:
                previous_value += walk
            else:
                previous_value -= walk
            values[i] = previous_value
        rw_time_series = TimeSeries(data[0])
        rw_time_series.add_values(values, (self.name, data))
        if previous_values is None:
            return rw_time_series, self.get_info(
                data, np.array(values[0 : data[0] - values_to_add])
            )
        else:
            return rw_time_series, self.get_info(data, np.array([previous_values[-1]]))


@hydra.main(
    version_base="1.2", config_path=os.path.join("..", ".."), config_name="config"
)
def show_plot(cfg: DictConfig):
    test_generator_linspace = LinspaceInfo(np.float64(0.0), np.float64(100.0), 100)
    method = RandomMethod(cfg.parameters_generation_method, test_generator_linspace)
    proc = SimpleRandomWalk(test_generator_linspace, method)
    test_sample = (100, proc.parameters_generator.generate_parameters())
    test_time_series, test_info = proc.generate_time_series(test_sample)
    print(test_time_series.get_values())
    draw_process_plot(test_time_series, test_info)


if __name__ == "__main__":
    show_plot()
