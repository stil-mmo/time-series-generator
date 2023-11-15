"""This module implements ProcessList class"""

from random import choice

from src.main.process import Process


class ProcessList:
    """This class provides methods to store processes and work with them"""

    def __init__(self):
        self.processes = {}
        self.num_processes = 0

    def add_processes(self, processes: list[Process]) -> None:
        """Adds processes to the process list"""
        for process in processes:
            if process.name not in self.processes:
                self.processes[process.name] = process
                self.num_processes += 1

    def remove_processes(self, process_names: list[str]) -> None:
        """Removes processes from the process list"""
        for process_name in process_names:
            if process_name in self.processes:
                self.processes.pop(process_name)
                self.num_processes -= 1
            else:
                raise ValueError(f"Process {process_name} not found in process list.")

    def get_processes(self, process_names: list[str]) -> list[Process]:
        """Returns processes from the process list"""
        processes = []
        for process_name in process_names:
            if process_name in self.processes:
                processes.append(self.processes[process_name])
            else:
                raise ValueError(f"Process {process_name} not found in process list.")
        return processes

    def get_random_processes(self, num_processes: int) -> list[Process]:
        """Returns random processes from the process list"""
        return [choice(list(self.processes.values())) for _ in range(num_processes)]

    def contains(self, process_name: str) -> bool:
        """Returns True if process list contains process with given name"""
        return process_name in self.processes
