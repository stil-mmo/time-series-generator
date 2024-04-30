from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class ParameterType(ABC):
    def __init__(
        self,
        constraints: NDArray[np.float64] = np.array([]),
        source_value: float = 0.0,
    ):
        self._source_value = source_value
        self._constraints = constraints

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def source_value(self) -> float:
        return self._source_value

    @source_value.setter
    def source_value(self, source_value: float) -> None:
        self._source_value = source_value

    @property
    def constraints(self) -> NDArray[np.float64]:
        return self._constraints

    @constraints.setter
    def constraints(self, constraints: NDArray[np.float64]) -> None:
        self._constraints = constraints


class StdType(ParameterType):
    def __init__(self, source_value: float = 0.0):
        super().__init__(source_value=source_value)

    @property
    def name(self) -> str:
        return "std_type"


class MeanType(ParameterType):
    def __init__(self, source_value: float = 0.0):
        super().__init__(source_value=source_value)

    @property
    def name(self) -> str:
        return "mean_type"


class CoefficientType(ParameterType):
    def __init__(
        self,
        constraints: NDArray[np.float64] = np.array([]),
        source_value: float = 0.0,
    ):
        super().__init__(constraints=constraints, source_value=source_value)

    @property
    def name(self) -> str:
        return "coefficient_type"
