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
from tsg.utils.typing import NDArrayFloat64T


class RandomMethod(ParametersGenerationMethod):
    name = "random_method"

    def __init__(
        self,
        linspace_info: LinspaceInfo,
    ) -> None:
        super().__init__(
            linspace_info=linspace_info,
        )

    def change_source_data(
        self,
        source_data: NDArrayFloat64T,
        parameters_required: list[ParameterType],
    ) -> NDArrayFloat64T:
        return np.zeros(shape=(1, len(parameters_required)))[0]

    def generate_std(self, std_type: StdType) -> float:
        return self.linspace_info.generate_std()

    def generate_mean(self, mean_type: MeanType) -> float:
        return self.linspace_info.generate_values(is_normal=False)[0]

    def generate_coefficient(self, coefficient_type: CoefficientType) -> float:
        constraints = coefficient_type.constraints
        return np.random.uniform(constraints[0], constraints[1])
