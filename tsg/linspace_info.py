import numpy as np
from numpy.typing import NDArray


class LinspaceInfo:
    def __init__(self, start: np.float64, stop: np.float64, parts: int = 50):
        self.start = start
        self.stop = stop
        self.parts = parts
        self.step = np.float64((stop - start) / parts)

    def generate_values(
        self, num_values=1, is_normal=True, center_shift=1.0
    ) -> NDArray[np.float64]:
        if is_normal:
            center = (self.start + self.stop) / 2
            return np.random.normal(center_shift * center, self.step, num_values)
        return np.random.uniform(self.start, self.stop, num_values)

    def generate_std(self, source_value: float | None = None) -> np.float64:
        k = np.float64(np.random.normal(0, self.step / 2))
        if k <= -self.step:
            k = np.float64(0)
        if source_value is None:
            std = np.float64(self.step + k)
        else:
            std = np.float64(self.step * (1 + source_value) + k)
        return std
