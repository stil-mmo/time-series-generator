from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo


class ParametersGenerationMethod(ABC):
    def __init__(
            self,
            parameters_generation_cfg: DictConfig,
            linspace_info: LinspaceInfo,
            source_data: NDArray[np.float64],
            parameters: dict[str: NDArray | None]
    ):
        self.parameters_generation_cfg = parameters_generation_cfg
        self.linspace_info = linspace_info
        self.source_data = source_data
        self.parameters = parameters

    @abstractmethod
    def generate_std(self) -> np.float64:
        pass

    @abstractmethod
    def generate_mean(self) -> np.float64:
        pass

    @abstractmethod
    def generate_coefficients(self) -> NDArray[np.float64]:
        pass
