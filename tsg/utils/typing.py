import numpy as np
from numpy.typing import NDArray

NDArrayFloat64 = NDArray[np.float64]
ParametersStepsType = tuple[int, NDArrayFloat64]
ProcessParametersType = list[ParametersStepsType]
ProcessOrderType = list[tuple[int, str]]
ProcessDataType = tuple[str, ProcessParametersType]
ProcessConfigType = tuple[str, ParametersStepsType]
