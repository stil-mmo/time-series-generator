import numpy as np
from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.process.process_storage import ProcessStorage
from tsg.scheduler.scheduler import Scheduler
from tsg.scheduler.scheduler_storage import SchedulerStorage
from tsg.time_series import TimeSeries
from tsg.utils.typing import NDArrayFloat64, ProcessDataType


class TimeSeriesGenerator:
    def __init__(
        self,
        cfg: DictConfig,
        linspace_info: LinspaceInfo,
        process_storage: ProcessStorage,
        scheduler_storage: SchedulerStorage | None = None,
    ):
        self.cfg = cfg
        self.ts_number = cfg.generation.ts_number
        self.ts_size = cfg.generation.ts_size
        self.linspace_info = linspace_info
        self.process_storage = process_storage
        self.scheduler_storage = scheduler_storage
        self.single_schedule = cfg.scheduler.single_schedule

    def generate_all(
        self,
    ) -> tuple[NDArrayFloat64, list[TimeSeries]]:
        ts_array: NDArrayFloat64 = np.ndarray((self.ts_number, self.ts_size))
        ts_list = []
        scheduler = self.generate_new_scheduler()
        iterations = (
            self.ts_number
            if self.scheduler_storage is None
            else self.scheduler_storage.points.shape[0]
        )
        for i in range(iterations):
            if self.scheduler_storage is None:
                source_data = None
                if not self.single_schedule:
                    scheduler = self.generate_new_scheduler()
                schedule = scheduler.generate_schedule()
            else:
                source_data = self.scheduler_storage.points[i]
                schedule = self.get_point_schedule(i)
            ts = self.generate_time_series(
                process_storage=self.process_storage,
                schedule=schedule,
                source_data=source_data,
            )
            ts_array[i] = ts.get_values()
            ts_list.append(ts)
        return ts_array, ts_list

    def generate_time_series(
        self,
        process_storage: ProcessStorage,
        schedule: list[ProcessDataType],
        source_data: NDArrayFloat64 | None = None,
    ) -> TimeSeries:
        current_time_series = TimeSeries(self.ts_size)
        for process_name, process_schedule in schedule:
            process = process_storage.get_processes([process_name])[0]
            for process_data in process_schedule:
                if current_time_series.last_index == 0:
                    current_time_series.add_values(
                        process.generate_time_series(
                            data=process_data,
                            source_data=source_data,
                        )[0].get_values(),
                        (process.name, process_data),
                    )
                else:
                    current_time_series.add_values(
                        process.generate_time_series(
                            process_data,
                            previous_values=current_time_series.get_values(),
                            source_data=source_data,
                        )[0].get_values(),
                        (process.name, process_data),
                    )
        return current_time_series

    def generate_new_scheduler(self) -> Scheduler:
        return Scheduler(
            num_steps=self.ts_size,
            linspace_info=self.linspace_info,
            process_storage=self.process_storage,
            strict_num_parts=self.cfg.scheduler.strict_num_parts,
            stable_parameters=self.cfg.scheduler.stable_parameters,
            process_order=self.cfg.scheduler.process_order,
        )

    def get_point_schedule(self, point_index: int) -> list[ProcessDataType]:
        if self.scheduler_storage is not None:
            cluster = self.scheduler_storage.get_cluster(point_index)
            scheduler = self.scheduler_storage.get_scheduler(cluster=cluster)
            schedule = scheduler.generate_schedule(
                source_data=self.scheduler_storage.points[point_index]
            )
        else:
            scheduler = self.generate_new_scheduler()
            schedule = scheduler.generate_schedule()
        return schedule
