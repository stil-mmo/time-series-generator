import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.parameters_generation.parameter_types import CoefficientType, StdType, MeanType
from tsg.parameters_generation.parameters_generation_method import ParametersGenerationMethod


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
        weights = self.calculate_weights(source_data.shape[0]) if \
            parameters_generation_cfg.aggregation_method.weighted_values else None
        self.mean_value = np.average(source_data, weights=weights)
        self.fraction = (
            self.mean_value / np.max(source_data)
            if parameters_generation_cfg.aggregation_method.use_max
            else self.mean_value / source_data.sum()
        )

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

    @staticmethod
    def calculate_weights(num_values: int) -> NDArray[np.float64]:
        progression_sum = (1 + num_values) * num_values / 2
        values = np.array([i + 1 for i in range(num_values)])
        return np.flip(values) / progression_sum
