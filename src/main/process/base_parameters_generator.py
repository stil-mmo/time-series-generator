from abc import ABC, abstractmethod

from numpy._typing import NDArray
from src.main.generator_linspace import GeneratorLinspace
from src.main.source_data_processing.aggregated_data import AggregatedData


class BaseParametersGenerator(ABC):
    def __init__(
        self,
        lag: int,
        generator_linspace: GeneratorLinspace,
        aggregated_data: AggregatedData | None = None,
    ):
        self.lag = lag
        self.generator_linspace = generator_linspace
        self.aggregated_data = aggregated_data

    @abstractmethod
    def generate_parameters(self) -> tuple[float, ...]:
        pass

    @abstractmethod
    def generate_init_values(self) -> NDArray:
        pass
