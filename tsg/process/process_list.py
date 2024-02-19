import numpy as np

from tsg.linspace_info import LinspaceInfo
from tsg.process.double_exponential_smoothing import DoubleExponentialSmoothing
from tsg.process.process import Process
from tsg.process.random_walk import RandomWalk
from tsg.process.simple_exponential_smoothing import SimpleExponentialSmoothing
from tsg.process.simple_random_walk import SimpleRandomWalk
from tsg.process.triple_exponential_smoothing import TripleExponentialSmoothing
from tsg.process.white_noise import WhiteNoise


class ProcessList:
    def __init__(self):
        self.processes = {}
        self.num_processes = 0

    def add_processes(self, processes: list[Process]) -> None:
        for process in processes:
            if process.name not in self.processes.keys():
                self.processes[process.name] = process
                self.num_processes += 1

    def add_all_processes(self, linspace_info: LinspaceInfo) -> None:
        self.add_processes(
            [
                WhiteNoise(linspace_info=linspace_info),
                SimpleRandomWalk(linspace_info=linspace_info),
                RandomWalk(linspace_info=linspace_info),
                SimpleExponentialSmoothing(linspace_info=linspace_info),
                DoubleExponentialSmoothing(linspace_info=linspace_info),
                TripleExponentialSmoothing(linspace_info=linspace_info, lag=12),
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
