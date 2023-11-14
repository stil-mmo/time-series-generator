from math import sqrt
from random import randint, shuffle

from src.main.process_list import ProcessList
from src.main.specific_processes.random_walk_process import RandomWalkProcess
from src.main.specific_processes.white_noise_process import WhiteNoiseProcess


class Scheduler:
    def __init__(
        self,
        num_steps: int,
        process_list: ProcessList = None,
        process_order: list[tuple[int, str]] = None,
    ):
        self.num_steps = num_steps
        self.process_list = (
            process_list if process_list is not None else self.generate_process_list()
        )
        self.process_order = (
            process_order
            if process_order is not None
            else self.generate_process_order()
        )

    @staticmethod
    def generate_process_list():
        process_list = ProcessList()
        process_list.add_processes([WhiteNoiseProcess(), RandomWalkProcess()])
        return process_list

    @staticmethod
    def generate_steps_number(
        num_max: int, num_parts: int, strict_num_parts: bool = False
    ) -> list[int]:
        current_num_max = num_max
        if num_parts <= 1:
            return [current_num_max]
        if strict_num_parts:
            steps_list = [num_max // num_parts] * num_parts
            steps_list[-1] += num_max % num_parts
            shuffle(steps_list)
            return steps_list
        steps_list = []
        while current_num_max > 0 and len(steps_list) < num_parts:
            if len(steps_list) == num_parts - 1:
                steps = current_num_max
            else:
                steps = randint(1, current_num_max)
            current_num_max -= steps
            steps_list.append(steps)
        shuffle(steps_list)
        return steps_list

    def generate_process_order(self) -> list[tuple[int, str]]:
        process_schedule = []
        num_parts = randint(1, int(sqrt(self.num_steps)))
        processes_steps = self.generate_steps_number(self.num_steps, num_parts)
        actual_num_parts = len(processes_steps)
        random_processes = self.process_list.get_random_processes(actual_num_parts)
        for i in range(actual_num_parts):
            process_schedule.append((processes_steps[i], random_processes[i].name))
        return process_schedule

    def generate_schedule(
        self, low_value: float, high_value: float, stable_parameters: bool = False
    ) -> list[tuple[str, list[tuple[int, tuple[float, ...]]]]]:
        schedule = []
        for steps, process_name in self.process_order:
            process = self.process_list.get_processes([process_name])[0]
            process_data = (process_name, [])
            if stable_parameters or steps == 1:
                process_data[1].append(
                    (steps, process.generate_parameters(low_value, high_value))
                )
            else:
                parameters_steps = self.generate_steps_number(steps, randint(1, steps))
                num_parts = len(parameters_steps)
                for i in range(num_parts):
                    process_data[1].append(
                        (
                            parameters_steps[i],
                            process.generate_parameters(low_value, high_value),
                        )
                    )
            schedule.append(process_data)
        return schedule
