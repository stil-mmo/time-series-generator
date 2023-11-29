from generator_linspace import GeneratorLinspace
from numpy import ndarray
from scheduler import Scheduler
from src.main.generator_typing import ProcessDataType, ProcessOrderType
from src.main.point_clustering import cluster_points
from src.main.point_sampling import move_points, sample_points
from src.main.process_list import ProcessList
from src.main.scheduler_storage import SchedulerStorage
from src.main.time_series import TimeSeries
from src.main.utils.utils import show_plot


class TimeSeriesGenerator:
    def __init__(
        self,
        num_time_series: int,
        num_steps: int,
        generator_linspace: GeneratorLinspace,
        scheduler_storage: SchedulerStorage,
        process_list: ProcessList | None = None,
        process_order: ProcessOrderType | None = None,
        stable_parameters: bool = True,
        single_schedule: bool = True,
    ):
        self.num_time_series = num_time_series
        self.num_steps = num_steps
        self.generator_linspace = generator_linspace
        self.scheduler_storage = scheduler_storage
        self.process_list = process_list
        self.process_order = process_order
        self.stable_parameters = stable_parameters
        self.single_schedule = single_schedule

    def generate_new_scheduler(self) -> Scheduler:
        return Scheduler(
            num_steps=self.num_steps,
            generator_linspace=self.generator_linspace,
            process_list=self.process_list,
            process_order=self.process_order,
        )

    def get_point_schedule(
        self, point_index: int
    ) -> tuple[ProcessList, list[ProcessDataType]]:
        cluster = self.scheduler_storage.get_cluster(point_index)
        scheduler = self.scheduler_storage.get_scheduler(cluster)
        scheduler.set_source_values(self.scheduler_storage.points[point_index])
        return scheduler.process_list, scheduler.generate_schedule(
            self.stable_parameters
        )

    def generate_time_series(
        self,
        process_list: ProcessList,
        schedule: list[ProcessDataType],
    ) -> TimeSeries:
        current_time_series = TimeSeries(self.num_steps)
        for process_name, process_schedule in schedule:
            process = process_list.get_processes([process_name])[0]
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

    def generate_all(
        self,
    ) -> tuple[ndarray, list[TimeSeries]]:
        ts_array = ndarray((self.num_time_series, self.num_steps))
        ts_list = []
        scheduler = self.generate_new_scheduler()
        for i in range(self.num_time_series):
            if self.scheduler_storage is None:
                if self.single_schedule:
                    ts = self.generate_time_series(
                        scheduler.process_list,
                        scheduler.generate_schedule(
                            stable_parameters=self.stable_parameters
                        ),
                    )
                else:
                    scheduler = self.generate_new_scheduler()
                    ts = self.generate_time_series(
                        scheduler.process_list,
                        scheduler.generate_schedule(
                            stable_parameters=self.stable_parameters
                        ),
                    )
                ts_array[i] = ts.get_values()
                ts_list.append(ts)
            else:
                process_list, schedule = self.get_point_schedule(i)
                ts = self.generate_time_series(
                    process_list,
                    schedule,
                )
                ts_array[i] = ts.get_values()
                ts_list.append(ts)
        return ts_array, ts_list


if __name__ == "__main__":
    coordinates, border_values = sample_points(5)
    coordinates = move_points(coordinates)
    print(coordinates)
    clusters = cluster_points(coordinates, 4)
    print(f"Clusters: {clusters}")
    test_generator_linspace = GeneratorLinspace(
        start=border_values[0], stop=border_values[1], parts=100
    )
    storage = SchedulerStorage(
        num_steps=100,
        generator_linspace=test_generator_linspace,
        points=coordinates,
        clusters=clusters,
    )
    ts_generator = TimeSeriesGenerator(
        num_time_series=coordinates.shape[0],
        num_steps=100,
        generator_linspace=test_generator_linspace,
        scheduler_storage=storage,
        stable_parameters=False,
        single_schedule=False,
    )

    time_series_array, time_series_list = ts_generator.generate_all()
    show_plot(time_series_array)
