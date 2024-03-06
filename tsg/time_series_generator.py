import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.process.process_storage import ProcessStorage
from tsg.sampling.aggregated_data import AggregatedData
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
        self.process_list = cfg.scheduler.process_list
        self.process_order = cfg.scheduler.process_order
        self.stable_parameters = cfg.generation.stable_parameters
        self.single_schedule = cfg.generation.single_schedule

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
                if not self.single_schedule:
                    scheduler = self.generate_new_scheduler()
                ts = self.generate_time_series(
                    scheduler.process_storage,
                    scheduler.generate_schedule(),
                )
            else:
                aggregated_data = AggregatedData(
                    source_data=self.scheduler_storage.points[i],
                    weighted_values=self.cfg.aggregated_data.weighted_values,
                    use_max=self.cfg.aggregated_data.use_max,
                )
                process_list, schedule = self.get_point_schedule(i, aggregated_data)
                ts = self.generate_time_series(
                    process_list=process_list,
                    schedule=schedule,
                    aggregated_data=aggregated_data,
                )
            ts_array[i] = ts.get_values()
            ts_list.append(ts)
        return ts_array, ts_list

    def generate_time_series(
        self,
        process_list: ProcessStorage,
        schedule: list[ProcessDataType],
        aggregated_data: AggregatedData | None = None,
    ) -> TimeSeries:
        current_time_series = TimeSeries(self.ts_size)
        for process_name, process_schedule in schedule:
            process = process_list.get_processes([process_name])[0]
            process.aggregated_data = aggregated_data
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
            ts_size=self.ts_size,
            linspace_info=self.linspace_info,
            process_list=self.process_list,
            process_order=self.process_order,
        )

    def get_point_schedule(
        self, point_index: int, aggregated_data: AggregatedData
    ) -> tuple[ProcessStorage, list[ProcessDataType]]:
        if self.scheduler_storage is not None:
            cluster = self.scheduler_storage.get_cluster(point_index)
            scheduler = self.scheduler_storage.get_scheduler(
                cluster=cluster, aggregated_data=aggregated_data
            )
        else:
            scheduler = Scheduler(self.ts_size, self.linspace_info)
        return scheduler.process_storage, scheduler.generate_schedule()
