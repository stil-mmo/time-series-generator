from abc import ABC, abstractmethod

from numpy.typing import NDArray
from src.main.generator_linspace import GeneratorLinspace
from src.main.process.base_parameters_generator import BaseParametersGenerator
from src.main.source_data_processing.aggregated_data import AggregatedData
from src.main.time_series import TimeSeries


class Process(ABC):
    def __init__(
        self,
        lag: int,
        generator_linspace: GeneratorLinspace,
        parameters_generator: BaseParametersGenerator,
        aggregated_data: AggregatedData | None = None,
    ):
        self._lag = lag
        self._generator_linspace = generator_linspace
        self._parameters_generator = parameters_generator
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
    def lag(self) -> int:
        return self._lag

    @lag.setter
    def lag(self, lag: int) -> None:
        self._lag = lag

    @property
    def generator_linspace(self) -> GeneratorLinspace:
        return self._generator_linspace

    @generator_linspace.setter
    def generator_linspace(self, generator_linspace: GeneratorLinspace) -> None:
        self._generator_linspace = generator_linspace

    @property
    def parameters_generator(self) -> BaseParametersGenerator:
        return self._parameters_generator

    @parameters_generator.setter
    def parameters_generator(
        self, parameters_generator: BaseParametersGenerator
    ) -> None:
        self._parameters_generator = parameters_generator

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
        data: tuple[int, tuple[float, ...]],
        previous_values: NDArray | None = None,
    ) -> tuple[TimeSeries, dict]:
        pass

    def get_info(
        self,
        data: tuple[int, tuple[float, ...]],
        init_values: tuple[float, ...] | None = None,
    ) -> dict:
        info = {
            "name": self.name,
            "lag": self.lag,
            "initial_values": init_values,
            "steps": data[0],
            "parameters": data[1],
        }
        return info
