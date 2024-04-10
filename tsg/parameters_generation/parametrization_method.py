import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.parameters_generation.parameter_types import ParameterType, CoefficientType, StdType, MeanType
from tsg.parameters_generation.parameters_generation_method import ParametersGenerationMethod


class ParametrizationMethod(ParametersGenerationMethod):
    def __init__(
            self,
            parameters_generation_cfg: DictConfig,
            linspace_info: LinspaceInfo,
            source_data: NDArray[np.float64],
            parameters_required: list[ParameterType],
    ):
        if len(source_data) != len(parameters_required):
            new_source_data = self.match_parameters_number()
        else:
            new_source_data = source_data
        super().__init__(
            parameters_generation_cfg=parameters_generation_cfg,
            linspace_info=linspace_info,
            source_data=new_source_data,
            parameters_required=parameters_required,
        )
        self.generate_all_parameters(set_source_data=True)

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

    def match_parameters_number(self) -> NDArray[np.float64]:
        old_size = len(self.source_data)
        new_size = len(self.parameters_required)
        new_source_data = np.zeros(shape=(1, new_size))[0]
        for i in range(min(old_size, new_size)):
            new_source_data[i] = self.source_data[i]
        if old_size < new_size:
            for j in range(old_size, new_size):
                average = np.average(new_source_data[:j])
                new_source_data[j] = average
        else:
            for j in range(new_size, old_size):
                new_source_data[(j - new_size) % new_size] += self.source_data[j]
        return new_source_data
