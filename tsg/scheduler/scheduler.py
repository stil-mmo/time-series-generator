from math import sqrt

import numpy as np

from tsg.linspace_info import LinspaceInfo
from tsg.process.process_storage import ProcessStorage
from tsg.utils.typing import NDArrayFloat64T, ProcessDataT, ProcessOrderT


class Scheduler:
    def __init__(
        self,
        num_steps: int,
        linspace_info: LinspaceInfo,
        process_storage: ProcessStorage,
        strict_num_parts: bool = True,
        stable_parameters: bool = False,
        process_order: ProcessOrderT | None = None,
    ) -> None:
        self.num_steps = num_steps
        self.linspace_info = linspace_info
        self.strict_num_parts = strict_num_parts
        self.stable_parameters = stable_parameters
        self.process_storage = process_storage
        self.process_order = (
            process_order
            if process_order is not None
            else self.generate_process_order()
        )

    def generate_schedule(
        self, source_data: NDArrayFloat64T | None = None
    ) -> list[ProcessDataT]:
        schedule = []
        for steps, process_name in self.process_order:
            process = self.process_storage.get_processes([process_name])[0]
            process_data: ProcessDataT = (process_name, [])
            if self.stable_parameters or steps == 1:
                process_data[1].append(
                    (
                        steps,
                        process.parameters_generator.generate_parameters(
                            source_data=source_data
                        ),
                    )
                )
            else:
                parameters_steps = self.generate_steps_number(
                    steps, np.random.randint(1, steps)
                )
                num_parts = len(parameters_steps)
                for i in range(num_parts):
                    process_data[1].append(
                        (
                            parameters_steps[i],
                            process.parameters_generator.generate_parameters(
                                source_data=source_data
                            ),
                        )
                    )
            schedule.append(process_data)
        return schedule

    def generate_process_order(self) -> ProcessOrderT:
        process_schedule = []
        num_parts = np.random.randint(1, int(sqrt(self.num_steps)))
        processes_steps = self.generate_steps_number(
            self.num_steps, num_parts, self.strict_num_parts
        )
        actual_num_parts = len(processes_steps)
        random_processes = self.process_storage.get_random_processes(actual_num_parts)
        for i in range(actual_num_parts):
            process_schedule.append((processes_steps[i], random_processes[i].name))
        return process_schedule

    def set_process_order(self, process_order: ProcessOrderT) -> None:
        self.process_order = process_order

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
            return steps_list
        steps_list = []
        while current_num_max > 1 and len(steps_list) < num_parts:
            if len(steps_list) == num_parts - 1:
                steps = current_num_max
            else:
                steps = np.random.randint(1, current_num_max)
            current_num_max -= steps
            steps_list.append(steps)
        if sum(steps_list) < num_max:
            steps_list.append(num_max - sum(steps_list))
        return steps_list
