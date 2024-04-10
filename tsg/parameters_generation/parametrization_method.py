import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.parameters_generation.parameter_types import CoefficientType, StdType, MeanType
from tsg.parameters_generation.parameters_generation_method import ParametersGenerationMethod


class ParametrizationMethod(ParametersGenerationMethod):
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

    def name(self) -> str:
        return "parametrization_method"

    def generate_std(self, std_type: StdType) -> np.float64:
        source_value = std_type.source_value
        high_border = 10 ** (np.log10(abs(source_value)) + 1)
        return self.linspace_info.generate_std(source_value=(source_value / high_border))

    def generate_mean(self, mean_type: MeanType) -> np.float64:
        return self.generate_value_in_range(
            source_value=mean_type.source_value,
            start=self.linspace_info.start,
            stop=self.linspace_info.stop
        )

    def generate_coefficient(self, coefficient_type: CoefficientType) -> np.float64:
        return self.generate_value_in_range(
            source_value=coefficient_type.source_value,
            start=coefficient_type.constraints[0],
            stop=coefficient_type.constraints[1]
        )
