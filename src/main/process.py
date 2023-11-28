from abc import ABC, abstractmethod

from numpy.typing import NDArray
from src.main.generator_linspace import GeneratorLinspace
from src.main.time_series import TimeSeries


class Process(ABC):
    def __init__(self, lag: int, generator_linspace: GeneratorLinspace):
        self._lag = lag
        self._generator_linspace = generator_linspace

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

    @abstractmethod
    def calculate_data(
        self, source_values: NDArray | None = None
    ) -> tuple[tuple[float, ...], NDArray]:
        pass

    @abstractmethod
    def generate_parameters(self) -> tuple[float, ...]:
        pass

    @abstractmethod
    def generate_init_values(self) -> NDArray:
        pass

    @abstractmethod
    def generate_time_series(
        self,
        sample: tuple[int, tuple[float, ...]],
        previous_values: NDArray | None = None,
    ) -> tuple[TimeSeries, dict]:
        pass

    def get_info(
        self,
        sample: tuple[int, tuple[float, ...]],
        init_values: tuple[float, ...] | None = None,
    ) -> dict:
        info = {
            "name": self.name,
            "lag": self.lag,
            "initial_values": init_values,
            "steps": sample[0],
            "parameters": sample[1],
        }
        return info
