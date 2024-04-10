from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.parameters_generation.parameter_types import ParameterType, CoefficientType, StdType, MeanType


class ParametersGenerationMethod(ABC):
    def __init__(
            self,
            parameters_generation_cfg: DictConfig,
            linspace_info: LinspaceInfo,
            source_data: NDArray[np.float64],
    ):
        self.parameters_generation_cfg = parameters_generation_cfg
        self.linspace_info = linspace_info
        self.source_data = source_data

    def generate_all_parameters(
            self,
            parameters_required: list[ParameterType],
            set_source_data: bool = False
    ) -> NDArray[np.float64]:
        if set_source_data and len(self.source_data) != len(parameters_required):
            self.source_data = self.match_parameters_number(new_size=len(parameters_required))
        parameters = np.zeros(shape=(1, len(parameters_required)))[0]
        for i in range(len(parameters_required)):
            parameter_type = parameters_required[i]
            if set_source_data:
                parameter_type.source_value = self.source_data[i]
            parameters[i] = self.generation_functions[parameter_type.name](parameter_type)
        return parameters

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def generate_std(self, std_type: StdType) -> np.float64:
        pass

    @abstractmethod
    def generate_mean(self, mean_type: MeanType) -> np.float64:
        pass

    @abstractmethod
    def generate_coefficient(self, coefficient_type: CoefficientType) -> np.float64:
        pass

    @property
    def generation_functions(self):
        return {
            "std_type": self.generate_std,
            "mean_type": self.generate_mean,
            "coefficient_type": self.generate_coefficient
        }

    def match_parameters_number(self, new_size: int) -> NDArray[np.float64]:
        old_size = len(self.source_data)
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

    @staticmethod
    def generate_value_in_range(source_value: np.float64, start: np.float64, stop: np.float64) -> np.float64:
        high_border = 10 ** (np.log10(abs(source_value)) + 1)
        value = stop * (source_value / high_border)
        if value < start:
            value = stop - value
        return value
