from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from tsg.linspace_info import LinspaceInfo
from tsg.sampling.aggregated_data import AggregatedData
from tsg.time_series import TimeSeries


class ParametersGenerator(ABC):
    def __init__(
        self,
        lag: int,
        linspace_info: LinspaceInfo,
        aggregated_data: AggregatedData | None = None,
    ):
        self.lag = lag
        self.linspace_info = linspace_info
        self._aggregated_data = aggregated_data

    @property
    def aggregated_data(self) -> AggregatedData | None:
        return self._aggregated_data

    @aggregated_data.setter
    def aggregated_data(self, aggregated_data: AggregatedData | None) -> None:
        self._aggregated_data = aggregated_data

    @abstractmethod
    def generate_parameters(self) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def generate_init_values(self) -> NDArray[np.float64]:
        pass


class Process(ABC):
    def __init__(
        self,
        linspace_info: LinspaceInfo,
        aggregated_data: AggregatedData | None = None,
    ):
        self._linspace_info = linspace_info
        self._aggregated_data = aggregated_data

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def num_parameters(self) -> int:
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
    def aggregated_data(self) -> AggregatedData | None:
        return self._aggregated_data

    @aggregated_data.setter
    def aggregated_data(self, aggregated_data: AggregatedData | None) -> None:
        self.parameters_generator.aggregated_data = aggregated_data
        self._aggregated_data = aggregated_data

    @abstractmethod
    def generate_time_series(
        self,
        data: tuple[int, NDArray[np.float64]],
        previous_values: NDArray[np.float64] | None = None,
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
