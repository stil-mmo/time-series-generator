import numpy as np

from tsg.linspace_info import LinspaceInfo
from tsg.parameters_generation.parameter_types import (
    CoefficientType,
    MeanType,
    ParameterType,
    StdType,
)
from tsg.parameters_generation.parameters_generation_method import (
    ParametersGenerationMethod,
)
from tsg.utils.typing import NDArrayFloat64


class ParametrizationMethod(ParametersGenerationMethod):
    def __init__(
        self,
        linspace_info: LinspaceInfo,
    ):
        super().__init__(
            linspace_info=linspace_info,
        )

    @property
    def name(self) -> str:
        return "parametrization_method"

    def change_source_data(
        self,
        source_data: NDArrayFloat64,
        parameters_required: list[ParameterType],
    ) -> NDArrayFloat64:
        return self.match_parameters_number(source_data, len(parameters_required))

    def generate_std(self, std_type: StdType) -> float:
        source_value = std_type.source_value
        high_border = 10 ** (np.log10(abs(source_value)) + 1)
        return self.linspace_info.generate_std(
            source_value=(source_value / high_border)
        )

    def generate_mean(self, mean_type: MeanType) -> float:
        return self.generate_value_in_range(
            source_value=mean_type.source_value,
            start=self.linspace_info.start,
            stop=self.linspace_info.stop,
        )

    def generate_coefficient(self, coefficient_type: CoefficientType) -> float:
        return self.generate_value_in_range(
            source_value=coefficient_type.source_value,
            start=coefficient_type.constraints[0],
            stop=coefficient_type.constraints[1],
        )

    @staticmethod
    def match_parameters_number(
        source_data: NDArrayFloat64, new_size: int
    ) -> NDArrayFloat64:
        old_size = len(source_data)
        if old_size == new_size:
            return source_data
        new_source_data = np.zeros(shape=(1, new_size))[0]
        for i in range(min(old_size, new_size)):
            new_source_data[i] = source_data[i]
        if old_size < new_size:
            for j in range(old_size, new_size):
                average = np.average(new_source_data[:j])
                new_source_data[j] = average
        else:
            for j in range(new_size, old_size):
                new_source_data[(j - new_size) % new_size] += source_data[j]
        return new_source_data
