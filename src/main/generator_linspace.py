from dataclasses import dataclass


@dataclass
class GeneratorLinspace:
    def __init__(self, start: float, stop: float, parts: int = 50):
        self.start = start
        self.stop = stop
        self.parts = parts
        self.step = (stop - start) / parts
