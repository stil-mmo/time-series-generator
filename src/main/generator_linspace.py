import numpy as np
from numpy.typing import NDArray


class GeneratorLinspace:
    def __init__(self, start: float, stop: float, parts: int = 50):
        self.start = start
        self.stop = stop
        self.parts = parts
        self.step = (stop - start) / parts

    def generate_values(
        self, num_values=1, is_normal=True, center_shift=1.0
    ) -> NDArray:
        if is_normal:
            center = (self.start + self.stop) / 2
            return np.random.normal(center_shift * center, self.step, num_values)
        return np.random.uniform(self.start, self.stop, num_values)

    def generate_std(self, source_value: float | None = None) -> float:
        k = np.random.normal(0, self.step / 2)
        if k <= -self.step:
            k = 0
        if source_value is None:
            std = self.step + k
        else:
            std = self.step * (1 + source_value) + k
        return std
