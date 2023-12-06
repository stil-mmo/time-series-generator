from numpy.random import normal, uniform
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
            return normal(center_shift * center, self.step, num_values)
        return uniform(self.start, self.stop, num_values)

    def generate_std(self, source_value: float | None = None) -> float:
        if source_value is None:
            std = abs(normal(self.step, self.step))
        else:
            std = self.step * (1.0 + source_value)
        return std
