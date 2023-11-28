from numpy.random import normal, uniform
from numpy.typing import NDArray
from numpy import sqrt


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

    def generate_std(self, std_coefficient=1.0) -> float:
        return sqrt(abs(normal(self.step, std_coefficient * self.step)))

    def calculate_std(self, source_value: float) -> float:
        std_coefficient = source_value / self.stop
        if std_coefficient >= 0.5:
            std = self.step * (1 + (1 - std_coefficient) * 2)
        else:
            std = self.step * (1 - std_coefficient * 2)
        return abs(std)
