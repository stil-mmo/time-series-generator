from abc import ABC, abstractmethod

import numpy as np

from tsg.linspace_info import LinspaceInfo
from tsg.parameters_generation.parameter_types import (
    CoefficientType,
    MeanType,
    ParameterType,
    StdType,
)
from tsg.utils.typing import NDArrayFloat64T


class ParametersGenerationMethod(ABC):
    name = ""

    def __init__(
        self,
        linspace_info: LinspaceInfo,
    ):
        self.linspace_info = linspace_info

    def generate_all_parameters(
        self,
        parameters_required: list[ParameterType],
        source_data: NDArrayFloat64T | None = None,
    ) -> NDArrayFloat64T:
        parameters = np.zeros(shape=(1, len(parameters_required)))[0]
        new_source_data = parameters.copy()
        if source_data is not None:
            new_source_data = self.change_source_data(source_data, parameters_required)
        for i in range(len(parameters_required)):
            parameter_type = parameters_required[i]
            if source_data is not None:
                parameter_type.source_value = new_source_data[i]
            parameters[i] = self.generation_functions[parameter_type.name](
                parameter_type
            )
        return parameters

    @abstractmethod
    def change_source_data(
        self,
        source_data: NDArrayFloat64T,
        parameters_required: list[ParameterType],
    ) -> NDArrayFloat64T:
        pass

    @abstractmethod
    def generate_std(self, std_type: StdType) -> float:
        pass

    @abstractmethod
    def generate_mean(self, mean_type: MeanType) -> float:
        pass

    @abstractmethod
    def generate_coefficient(self, coefficient_type: CoefficientType) -> float:
        pass

    @property
    def generation_functions(self):
        return {
            "std_type": self.generate_std,
            "mean_type": self.generate_mean,
            "coefficient_type": self.generate_coefficient,
        }

    def get_mean_value(
        self, source_data: NDArrayFloat64T | None, weighted: bool = True
    ) -> float:
        if source_data is not None:
            weights = self.calculate_weights(len(source_data)) if weighted else None
            mean_value = float(np.average(source_data, weights=weights))
        else:
            mean_value = np.random.uniform(
                self.linspace_info.start, self.linspace_info.stop
            )
        return mean_value

    @staticmethod
    def generate_value_in_range(
        source_value: float, start: float, stop: float
    ) -> float:
        high_border = 10 ** (np.log10(abs(source_value)) + 1)
        value = stop * (source_value / high_border)
        if value < start:
            value = stop - value
        return value

    @staticmethod
    def calculate_weights(num_values: int) -> NDArrayFloat64T:
        progression_sum = (1 + num_values) * num_values / 2
        values = np.array([i + 1 for i in range(num_values)])
        return np.flip(values) / progression_sum
