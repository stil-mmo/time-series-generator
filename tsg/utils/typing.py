import numpy as np
from numpy.typing import NDArray

NDArrayFloat64T = NDArray[np.float64]
ParametersStepsT = tuple[int, NDArrayFloat64T]
ProcessParametersT = list[ParametersStepsT]
ProcessOrderT = list[tuple[int, str]]
ProcessDataT = tuple[str, ProcessParametersT]
ProcessConfigT = tuple[str, ParametersStepsT]
