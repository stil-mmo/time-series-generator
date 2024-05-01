import numpy as np
from numpy.typing import NDArray

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


class AggregationMethod(ParametersGenerationMethod):
    def __init__(
        self,
        linspace_info: LinspaceInfo,
        weighted_values: bool = True,
        use_max: bool = False,
    ):
        self.weighted_values = weighted_values
        self.use_max = use_max

        super().__init__(
            linspace_info=linspace_info,
        )

    @property
    def name(self) -> str:
        return "aggregation_method"

    def change_source_data(
        self,
        source_data: NDArray[np.float64],
        parameters_required: list[ParameterType],
    ) -> NDArray[np.float64]:
        new_source_data = np.zeros(shape=(1, len(parameters_required)))[0]
        mean_value = self.get_mean_value(source_data, self.weighted_values)
        fraction = (
            mean_value / np.max(source_data)
            if self.use_max
            else mean_value / source_data.sum()
        )
        for i in range(len(new_source_data)):
            if parameters_required[i].name in ("std_type", "coefficient_type"):
                new_source_data[i] = fraction
            else:
                new_source_data[i] = mean_value
        return new_source_data

    def generate_std(self, std_type: StdType) -> float:
        return self.linspace_info.generate_std(source_value=std_type.source_value)

    def generate_mean(self, mean_type: MeanType) -> float:
        return mean_type.source_value

    def generate_coefficient(self, coefficient_type: CoefficientType) -> float:
        coefficient = coefficient_type.constraints[1] * coefficient_type.source_value
        if coefficient < coefficient_type.constraints[0]:
            coefficient = coefficient_type.constraints[1] - coefficient
        return coefficient
