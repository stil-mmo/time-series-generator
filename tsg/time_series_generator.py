import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.parameters_generation.all_generation_methods import ALL_GENERATION_METHODS
from tsg.parameters_generation.parameters_generation_method import (
    ParametersGenerationMethod,
)
from tsg.parameters_generation.random_method import RandomMethod
from tsg.process.process_storage import ProcessStorage
from tsg.scheduler.scheduler import Scheduler
from tsg.scheduler.scheduler_storage import SchedulerStorage
from tsg.time_series import TimeSeries
from tsg.utils.typing import ProcessDataType


class TimeSeriesGenerator:
    def __init__(
        self,
        cfg: DictConfig,
        linspace_info: LinspaceInfo,
        scheduler_storage: SchedulerStorage | None = None,
    ):
        self.cfg = cfg
        self.ts_number = cfg.generation.ts_number
        self.ts_size = cfg.generation.ts_size
        self.linspace_info = linspace_info
        self.scheduler_storage = scheduler_storage
        self.single_schedule = cfg.scheduler.single_schedule
        self.generation_method_name = cfg.generation.generation_method

    def generate_all(
        self,
    ) -> tuple[NDArray[np.float64], list[TimeSeries]]:
        ts_array: NDArray[np.float64] = np.ndarray((self.ts_number, self.ts_size))
        ts_list = []
        scheduler = self.generate_new_scheduler()
        iterations = (
            self.ts_number
            if self.scheduler_storage is None
            else self.scheduler_storage.points.shape[0]
        )
        for i in range(iterations):
            if self.scheduler_storage is None:
                generation_method = RandomMethod(
                    parameters_generation_cfg=self.cfg.parameters_generation_method,
                    linspace_info=self.linspace_info,
                )
                if not self.single_schedule:
                    scheduler = self.generate_new_scheduler()
                process_list = scheduler.process_storage
                schedule = scheduler.generate_schedule()
            else:
                generation_method = ALL_GENERATION_METHODS[self.generation_method_name](
                    parameters_generation_cfg=self.cfg.parameters_generation_method,
                    linspace_info=self.linspace_info,
                    source_data=self.scheduler_storage.points[i],
                )
                process_list, schedule = self.get_point_schedule(i, generation_method)
            ts = self.generate_time_series(
                process_list=process_list,
                schedule=schedule,
                generation_method=generation_method,
            )
            ts_array[i] = ts.get_values()
            ts_list.append(ts)
        return ts_array, ts_list

    def generate_time_series(
        self,
        process_list: ProcessStorage,
        schedule: list[ProcessDataType],
        generation_method: ParametersGenerationMethod,
    ) -> TimeSeries:
        current_time_series = TimeSeries(self.ts_size)
        for process_name, process_schedule in schedule:
            process = process_list.get_processes([process_name])[0]
            process.parameters_generation_method = generation_method
            for process_data in process_schedule:
                if current_time_series.last_index == 0:
                    current_time_series.add_values(
                        process.generate_time_series(process_data)[0].get_values(),
                        (process.name, process_data),
                    )
                else:
                    current_time_series.add_values(
                        process.generate_time_series(
                            process_data,
                            previous_values=current_time_series.get_values(),
                        )[0].get_values(),
                        (process.name, process_data),
                    )
        return current_time_series

    def generate_new_scheduler(self) -> Scheduler:
        return Scheduler(
            cfg=self.cfg,
            linspace_info=self.linspace_info,
        )

    def get_point_schedule(
        self, point_index: int, generation_method: ParametersGenerationMethod
    ) -> tuple[ProcessStorage, list[ProcessDataType]]:
        if self.scheduler_storage is not None:
            cluster = self.scheduler_storage.get_cluster(point_index)
            scheduler = self.scheduler_storage.get_scheduler(
                cluster=cluster, parameters_generation_method=generation_method
            )
        else:
            scheduler = Scheduler(self.ts_size, self.linspace_info)
        return scheduler.process_storage, scheduler.generate_schedule()
