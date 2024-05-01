import numpy as np

from tsg.utils.typing import NDArrayFloat64T


class LinspaceInfo:
    def __init__(
        self,
        start: float,
        stop: float,
        parts: int = 50,
        center_shift: float = 1.0,
        step_coeff: float = 0.5,
        use_k: bool = True,
    ):
        self.start = start
        self.stop = stop
        self.parts = parts
        self.step = (stop - start) / parts
        self.center_shift = center_shift
        self.step_coeff = step_coeff
        self.use_k = use_k

    def generate_values(self, num_values=1, is_normal=True) -> NDArrayFloat64T:
        if is_normal:
            center = (self.start + self.stop) / 2
            return np.random.normal(self.center_shift * center, self.step, num_values)
        return np.random.uniform(self.start, self.stop, num_values)

    def generate_std(self, source_value: float | None = None) -> float:
        k = np.random.normal(0, self.step * self.step_coeff)
        if k <= -self.step:
            k = 0.0
        if source_value is None:
            std = self.step + k
        else:
            use_k = self.use_k
            std = self.step * (1 + source_value) + k * use_k
        return std
