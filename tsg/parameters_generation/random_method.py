import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.parameters_generation.parameter_types import CoefficientType, MeanType, StdType
from tsg.parameters_generation.parameters_generation_method import (
    ParametersGenerationMethod,
)


class RandomMethod(ParametersGenerationMethod):
    def __init__(
        self,
        parameters_generation_cfg: DictConfig,
        linspace_info: LinspaceInfo,
        source_data: NDArray[np.float64] = np.array([]),
    ):
        value = linspace_info.generate_values(is_normal=False)[0]
        super().__init__(
            parameters_generation_cfg=parameters_generation_cfg,
            linspace_info=linspace_info,
            source_data=source_data,
            mean_value=value,
        )

    @property
    def name(self) -> str:
        return "random_method"

    def generate_std(self, std_type: StdType) -> float:
        return self.linspace_info.generate_std()

    def generate_mean(self, mean_type: MeanType) -> float:
        return self.linspace_info.generate_values(is_normal=False)[0]

    def generate_coefficient(self, coefficient_type: CoefficientType) -> float:
        constraints = coefficient_type.constraints
        return np.random.uniform(constraints[0], constraints[1])
