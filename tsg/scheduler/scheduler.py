from math import sqrt

import numpy as np
from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.parameters_generation.parameters_generation_method import (
    ParametersGenerationMethod,
)
from tsg.process.process_storage import ProcessStorage
from tsg.utils.typing import ProcessDataType, ProcessOrderType


class Scheduler:
    def __init__(
        self,
        cfg: DictConfig,
        linspace_info: LinspaceInfo,
        parameters_generation_method: ParametersGenerationMethod | None = None,
    ):
        self.num_steps = cfg.generation.ts_size
        self.linspace_info = linspace_info
        self.strict_num_parts = cfg.scheduler.strict_num_parts
        self.stable_parameters = cfg.scheduler.stable_parameters
        self.process_storage = ProcessStorage(cfg=cfg, linspace_info=linspace_info)
        self.process_order = (
            cfg.scheduler.process_order
            if cfg.scheduler.process_order is not None
            else self.generate_process_order()
        )
        self.parameters_generation_method = parameters_generation_method

    def generate_schedule(self) -> list[ProcessDataType]:
        schedule = []
        for steps, process_name in self.process_order:
            process = self.process_storage.get_processes([process_name])[0]
            if self.parameters_generation_method is not None:
                process.parameters_generation_method = self.parameters_generation_method
            process_data: ProcessDataType = (process_name, [])
            if self.stable_parameters or steps == 1:
                process_data[1].append(
                    (steps, process.parameters_generator.generate_parameters())
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
                            process.parameters_generator.generate_parameters(),
                        )
                    )
            schedule.append(process_data)
        return schedule

    def generate_process_order(self) -> ProcessOrderType:
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

    def set_process_order(self, process_order: ProcessOrderType) -> None:
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
