import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from src.main.generator_linspace import GeneratorLinspace
from src.main.generator_typing import ProcessDataType, ProcessOrderType
from src.main.process.process_list import ProcessList
from src.main.scheduler.scheduler import Scheduler
from src.main.scheduler.scheduler_storage import SchedulerStorage
from src.main.source_data_processing.aggregated_data import AggregatedData
from src.main.source_data_processing.point_clustering import cluster_points
from src.main.source_data_processing.point_sampling import sample_points
from src.main.time_series import TimeSeries


class TimeSeriesGenerator:
    def __init__(
        self,
        num_time_series: int,
        num_steps: int,
        generator_linspace: GeneratorLinspace,
        scheduler_storage: SchedulerStorage | None = None,
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
        self, point_index: int, aggregated_data: AggregatedData
    ) -> tuple[ProcessList, list[ProcessDataType]]:
        cluster = self.scheduler_storage.get_cluster(point_index)
        scheduler = self.scheduler_storage.get_scheduler(
            cluster=cluster, aggregated_data=aggregated_data
        )
        return scheduler.process_list, scheduler.generate_schedule(
            stable_parameters=True
        )

    def generate_time_series(
        self,
        process_list: ProcessList,
        schedule: list[ProcessDataType],
        aggregated_data: AggregatedData | None = None,
    ) -> TimeSeries:
        current_time_series = TimeSeries(self.num_steps)
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

    def generate_all(
        self,
    ) -> tuple[NDArray, list[TimeSeries]]:
        ts_array = np.ndarray((self.num_time_series, self.num_steps))
        ts_list = []
        scheduler = self.generate_new_scheduler()
        iterations = (
            self.num_time_series
            if self.scheduler_storage is None
            else self.scheduler_storage.points.shape[0]
        )
        for i in range(iterations):
            if self.scheduler_storage is None:
                if not self.single_schedule:
                    scheduler = self.generate_new_scheduler()
                ts = self.generate_time_series(
                    scheduler.process_list,
                    scheduler.generate_schedule(
                        stable_parameters=self.stable_parameters
                    ),
                )
            else:
                aggregated_data = AggregatedData(
                    source_data=self.scheduler_storage.points[i]
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


def show_result(coordinates, clusters, border_values, shift, time_series_array):
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle("Generated time series with parameters clustering")

    ax = fig.add_subplot(1, 2, 1)
    colors = ["blue", "red", "green"]
    for i in range(5):
        time_series = time_series_array[i]
        ax.text(100, time_series[-1], f"TS {i}")
        ax.plot(time_series, color=colors[clusters[i]])
    ax.grid(True)
    ax.set_xlabel("time")
    ax.set_ylabel("values")

    # Second subplot
    ax = fig.add_subplot(1, 2, 2, projection="3d")

    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))
    x += shift
    y += shift
    z += shift

    surf = ax.plot_wireframe(x, y, z, color="grey", alpha=0.3)
    for i in range(5):
        ax.scatter(
            coordinates[i][0],
            coordinates[i][1],
            coordinates[i][2],
            color=colors[clusters[i]],
        )
    ax.set_zlim(border_values[0], border_values[1])
    plt.show()


if __name__ == "__main__":
    coordinates, border_values, shift = sample_points(5)
    clusters = cluster_points(coordinates, 2)
    print(coordinates)
    print(f"Clusters: {clusters}")
    print(shift)
    test_generator_linspace = GeneratorLinspace(
        start=border_values[0], stop=border_values[1], parts=100
    )
    print(
        test_generator_linspace.start,
        test_generator_linspace.stop,
        test_generator_linspace.step,
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
    pl = ProcessList()
    pl.add_all_processes(test_generator_linspace)
    time_series_array, time_series_list = ts_generator.generate_all()
    with open("logs/generation.txt", "w") as logs:
        logs.write("TS generation logs\n\n")
    for i in range(len(time_series_list)):
        ts = time_series_list[i]
        ts.dump_logs(pl, "logs/generation.txt", i)
    show_result(coordinates, clusters, border_values, shift, time_series_array)
