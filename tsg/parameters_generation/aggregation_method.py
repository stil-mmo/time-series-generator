import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.parameters_generation.parameter_types import CoefficientType, MeanType, StdType
from tsg.parameters_generation.parameters_generation_method import (
    ParametersGenerationMethod,
)


class AggregationMethod(ParametersGenerationMethod):
    def __init__(
        self,
        parameters_generation_cfg: DictConfig,
        linspace_info: LinspaceInfo,
        source_data: NDArray[np.float64],
    ):
        super().__init__(
            parameters_generation_cfg=parameters_generation_cfg,
            linspace_info=linspace_info,
            source_data=source_data,
        )

        self.fraction = (
            self.mean_value / np.max(source_data)
            if parameters_generation_cfg.aggregation_method.use_max
            else self.mean_value / source_data.sum()
        )

    @property
    def name(self) -> str:
        return "aggregation_method"

    def generate_std(self, std_type: StdType) -> np.float64:
        return self.linspace_info.generate_std(source_value=self.fraction)

    def generate_mean(self, mean_type: MeanType) -> np.float64:
        return self.mean_value

    def generate_coefficient(self, coefficient_type: CoefficientType) -> np.float64:
        coefficient = coefficient_type.constraints[1] * self.fraction
        if coefficient < coefficient_type.constraints[0]:
            coefficient = coefficient_type.constraints[1] - coefficient
        return coefficient
