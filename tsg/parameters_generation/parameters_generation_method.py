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
            parameters_required: list[ParameterType],
    ):
        self.parameters_generation_cfg = parameters_generation_cfg
        self.linspace_info = linspace_info
        self.source_data = source_data
        self.parameters_required = parameters_required
        self.parameters = np.zeros(shape=(1, len(parameters_required)))[0]

    def generate_all_parameters(self, set_source_data: bool = False):
        for i in range(len(self.parameters_required)):
            parameter_type = self.parameters_required[i]
            if set_source_data:
                parameter_type.source_value = self.source_data[i]
            self.parameters[i] = self.generation_functions[parameter_type.name](parameter_type)

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
            "std": self.generate_std,
            "mean": self.generate_mean,
            "coefficient_type": self.generate_coefficient
        }

    @staticmethod
    def generate_value_in_range(source_value: np.float64, start: np.float64, stop: np.float64) -> np.float64:
        high_border = 10 ** (np.log10(abs(source_value)) + 1)
        value = stop * (source_value / high_border)
        if value < start:
            value = stop - value
        return value

    def get_parameters(self):
        return self.parameters
