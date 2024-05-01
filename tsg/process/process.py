from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from tsg.linspace_info import LinspaceInfo
from tsg.parameters_generation.parameter_types import ParameterType
from tsg.parameters_generation.parameters_generation_method import (
    ParametersGenerationMethod,
)
from tsg.time_series import TimeSeries


class ParametersGenerator(ABC):
    def __init__(
        self,
        lag: int,
        linspace_info: LinspaceInfo,
        parameters_generation_method: ParametersGenerationMethod,
        parameters_required: list[ParameterType],
    ):
        self.lag = lag
        self.linspace_info = linspace_info
        self._parameters_generation_method = parameters_generation_method
        self.parameters_required = parameters_required

    @property
    def parameters_generation_method(self) -> ParametersGenerationMethod:
        return self._parameters_generation_method

    @parameters_generation_method.setter
    def parameters_generation_method(
        self, parameters_generation_method: ParametersGenerationMethod
    ) -> None:
        self._parameters_generation_method = parameters_generation_method

    @abstractmethod
    def generate_parameters(
        self, source_data: NDArray | None = None
    ) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def generate_init_values(
        self, source_data: NDArray | None = None
    ) -> NDArray[np.float64]:
        pass


class Process(ABC):
    def __init__(
        self,
        linspace_info: LinspaceInfo,
        parameters_generation_method: ParametersGenerationMethod,
    ):
        self._linspace_info = linspace_info
        self._parameters_generation_method = parameters_generation_method

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def parameters(self) -> list[ParameterType]:
        pass

    @property
    @abstractmethod
    def lag(self) -> int:
        pass

    @property
    @abstractmethod
    def parameters_generator(self) -> ParametersGenerator:
        pass

    @property
    def linspace_info(self) -> LinspaceInfo:
        return self._linspace_info

    @linspace_info.setter
    def linspace_info(self, linspace_info: LinspaceInfo) -> None:
        self._linspace_info = linspace_info

    @property
    def parameters_generation_method(self) -> ParametersGenerationMethod:
        return self._parameters_generation_method

    @parameters_generation_method.setter
    def parameters_generation_method(
        self, parameters_generation_method: ParametersGenerationMethod
    ) -> None:
        self.parameters_generator.parameters_generation_method = (
            parameters_generation_method
        )
        self._parameters_generation_method = parameters_generation_method

    @abstractmethod
    def generate_time_series(
        self,
        data: tuple[int, NDArray[np.float64]],
        previous_values: NDArray[np.float64] | None = None,
        source_data: NDArray[np.float64] | None = None,
    ) -> tuple[TimeSeries, dict]:
        pass

    def get_info(
        self,
        data: tuple[int, NDArray[np.float64]],
        init_values: NDArray[np.float64] | None = None,
    ) -> dict:
        info = {
            "name": self.name,
            "lag": self.lag,
            "initial_values": init_values,
            "steps": data[0],
            "parameters": data[1],
        }
        return info
