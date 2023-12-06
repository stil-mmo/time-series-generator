from random import choice

from src.main.process.process import Process


class ProcessList:
    def __init__(self):
        self.processes = {}
        self.num_processes = 0

    def add_processes(self, processes: list[Process]) -> None:
        for process in processes:
            if process.name not in self.processes.keys():
                self.processes[process.name] = process
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
        return [choice(list(self.processes.values())) for _ in range(num_processes)]

    def contains(self, process_name: str) -> bool:
        return process_name in self.processes.keys()
