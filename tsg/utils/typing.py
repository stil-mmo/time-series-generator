import numpy as np
from numpy.typing import NDArray

ParametersStepsType = tuple[int, NDArray[np.float64]]
ProcessParametersType = list[ParametersStepsType]
ProcessOrderType = list[tuple[int, str]]
ProcessDataType = tuple[str, ProcessParametersType]
ProcessConfigType = tuple[str, ParametersStepsType]
