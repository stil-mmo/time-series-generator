import numpy as np
from src.main.generator_linspace import GeneratorLinspace
from src.main.process.double_exponential_smoothing import DoubleExponentialSmoothing
from src.main.process.process import Process
from src.main.process.random_walk import RandomWalk
from src.main.process.simple_exponential_smoothing import SimpleExponentialSmoothing
from src.main.process.simple_random_walk import SimpleRandomWalk
from src.main.process.triple_exponential_smoothing import TripleExponentialSmoothing
from src.main.process.white_noise_process import WhiteNoiseProcess


class ProcessList:
    def __init__(self):
        self.processes = {}
        self.num_processes = 0

    def add_processes(self, processes: list[Process]) -> None:
        for process in processes:
            if process.name not in self.processes.keys():
                self.processes[process.name] = process
                self.num_processes += 1

    def add_all_processes(self, generator_linspace: GeneratorLinspace) -> None:
        self.add_processes(
            [
                WhiteNoiseProcess(generator_linspace=generator_linspace),
                SimpleRandomWalk(generator_linspace=generator_linspace),
                RandomWalk(generator_linspace=generator_linspace),
                SimpleExponentialSmoothing(generator_linspace=generator_linspace),
                DoubleExponentialSmoothing(generator_linspace=generator_linspace),
                TripleExponentialSmoothing(
                    generator_linspace=generator_linspace, lag=12
                ),
            ]
        )

    def remove_processes(self, process_names: list[str]) -> None:
        for process_name in process_names:
            if process_name in self.processes.keys():
                self.processes.pop(process_name)
                self.num_processes -= 1
            else:
                print(f"Process {process_name} not found in process list.")

    def get_processes(self, process_names: list[str]) -> list[Process]:
        processes = []
        for process_name in process_names:
            if process_name in self.processes.keys():
                processes.append(self.processes[process_name])
            else:
                print(f"Process {process_name} not found in process list.")
        return processes

    def get_random_processes(self, num_processes: int) -> list[Process]:
        return [
            np.random.choice(list(self.processes.values()))
            for _ in range(num_processes)
        ]

    def contains(self, process_name: str) -> bool:
        return process_name in self.processes.keys()
