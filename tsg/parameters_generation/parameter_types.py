from abc import ABC

import numpy as np

from tsg.utils.typing import NDArrayFloat64T


class ParameterType(ABC):
    name = ""

    def __init__(
        self,
        constraints: NDArrayFloat64T = np.array([0.0, 1.0]),
        source_value: float = 0.0,
    ) -> None:
        self._source_value = source_value
        self._constraints = constraints

    @property
    def source_value(self) -> float:
        return self._source_value

    @source_value.setter
    def source_value(self, source_value: float) -> None:
        self._source_value = source_value

    @property
    def constraints(self) -> NDArrayFloat64T:
        return self._constraints

    @constraints.setter
    def constraints(self, constraints: NDArrayFloat64T) -> None:
        self._constraints = constraints


class StdType(ParameterType):
    name = "std_type"

    def __init__(self, source_value: float = 0.0) -> None:
        super().__init__(source_value=source_value)


class MeanType(ParameterType):
    name = "mean_type"

    def __init__(self, source_value: float = 0.0) -> None:
        super().__init__(source_value=source_value)


class CoefficientType(ParameterType):
    name = "coefficient_type"

    def __init__(
        self,
        constraints: NDArrayFloat64T = np.array([0.0, 1.0]),
        source_value: float = 0.0,
    ) -> None:
        super().__init__(constraints=constraints, source_value=source_value)
