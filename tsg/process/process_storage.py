from random import choice

from hydra.utils import instantiate
from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.parameters_generation.parameters_generation_method import (
    ParametersGenerationMethod,
)
from tsg.process.double_exponential_smoothing import DoubleExponentialSmoothing
from tsg.process.process import Process
from tsg.process.random_walk import RandomWalk
from tsg.process.simple_exponential_smoothing import SimpleExponentialSmoothing
from tsg.process.simple_random_walk import SimpleRandomWalk
from tsg.process.triple_exponential_smoothing import TripleExponentialSmoothing
from tsg.process.white_noise import WhiteNoise

ALL_PROCESSES = {
    "white_noise": WhiteNoise,
    "simple_random_walk": SimpleRandomWalk,
    "random_walk": RandomWalk,
    "simple_exponential_smoothing": SimpleExponentialSmoothing,
    "double_exponential_smoothing": DoubleExponentialSmoothing,
    "triple_exponential_smoothing": TripleExponentialSmoothing,
}


class ProcessStorage:
    def __init__(
        self,
        cfg_process: DictConfig,
        linspace_info: LinspaceInfo,
        generation_method: ParametersGenerationMethod,
        process_list: list[str] | None = None,
    ) -> None:
        self.cfg_process = cfg_process
        self.linspace_info = linspace_info
        self.processes: dict[str, Process] = {}
        self.num_processes = 0
        self.generation_method = generation_method
        if process_list is not None:
            self.add_processes(process_list)
        else:
            self.add_processes(list(ALL_PROCESSES.keys()))

    def add_processes(self, process_list: list[str]) -> None:
        for process_name in process_list:
            if process_name not in self.processes.keys():
                if self.cfg_process.get(process_name) is not None:
                    process = instantiate(
                        self.cfg_process[process_name],
                        linspace_info=self.linspace_info,
                        parameters_generation_method=self.generation_method,
                    )
                else:
                    process = ALL_PROCESSES[process_name](
                        linspace_info=self.linspace_info,
                        parameters_generation_method=self.generation_method,
                    )
                self.processes[process_name] = process
                self.num_processes += 1

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
        processes: list[Process] = list(self.processes.values())
        return [choice(processes) for _ in range(num_processes)]

    def contains(self, process_name: str) -> bool:
        return process_name in self.processes.keys()
