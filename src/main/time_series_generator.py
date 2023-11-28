from generator_linspace import GeneratorLinspace
from numpy import array, ndarray
from numpy.typing import NDArray
from scheduler import PROCESS_ORDER, PROCESS_SAMPLES, Scheduler
from src.main.point_sampling import sample_points
from src.main.process_list import ProcessList
from src.main.time_series import TimeSeries
from src.main.utils.utils import show_plot


class TimeSeriesGenerator:
    def __init__(
        self,
        num_time_series: int,
        num_steps: int,
        process_list: ProcessList | None = None,
        process_order: PROCESS_ORDER | None = None,
        points: NDArray | None = None,
    ):
        self.num_time_series = num_time_series
        self.num_steps = num_steps
        self.process_list = process_list
        self.process_order = process_order
        self.points = points

    def generate_time_series(
        self,
        process_list: ProcessList,
        schedule: list[PROCESS_SAMPLES],
    ) -> TimeSeries:
        current_time_series = TimeSeries(self.num_steps)
        for process_name, process_schedule in schedule:
            process = process_list.get_processes([process_name])[0]
            for sample in process_schedule:
                if current_time_series.last_index == 0:
                    current_time_series.add_values(
                        process.generate_time_series(sample)[0].get_values(),
                        (process.name, sample),
                    )
                else:
                    current_time_series.add_values(
                        process.generate_time_series(
                            sample, previous_values=current_time_series.get_values()
                        )[0].get_values(),
                        (process.name, sample),
                    )
        return current_time_series

    def generate_all(
        self,
        generator_linspace: GeneratorLinspace,
        stable_parameters: bool = True,
        single_schedule: bool = True,
    ) -> tuple[ndarray, list[TimeSeries]]:
        time_series_array = ndarray((self.num_time_series, self.num_steps))
        time_series_list = []
        scheduler = Scheduler(
            num_steps=self.num_steps,
            generator_linspace=generator_linspace,
            process_list=self.process_list,
            process_order=self.process_order,
        )
        if self.points is not None:
            scheduler.set_source_values(self.points[0])
        schedule = (
            None
            if not single_schedule
            else scheduler.generate_schedule(stable_parameters=stable_parameters)
        )
        for i in range(self.num_time_series):
            if schedule is None:
                if self.points is None:
                    time_series = self.generate_time_series(
                        scheduler.process_list,
                        scheduler.generate_schedule(
                            stable_parameters=stable_parameters
                        ),
                    )
                    time_series_array[i] = time_series.get_values()
                    time_series_list.append(time_series)
                else:
                    scheduler.set_source_values(self.points[i])
                    time_series = self.generate_time_series(
                        scheduler.process_list,
                        scheduler.generate_schedule(
                            stable_parameters=stable_parameters
                        ),
                    )
                    time_series_array[i] = time_series.get_values()
                    time_series_list.append(time_series)
            else:
                time_series = self.generate_time_series(
                    scheduler.process_list, schedule
                )
                time_series_array[i] = time_series.get_values()
                time_series_list.append(time_series)
        return time_series_array, time_series_list


if __name__ == "__main__":
    coordinates, border_values = sample_points(3)
    ts_generator = TimeSeriesGenerator(
        num_time_series=3, num_steps=100, points=coordinates
    )
    test_generator_linspace = GeneratorLinspace(
        start=border_values[0], stop=border_values[1], parts=100
    )
    time_series_array, time_series_list = ts_generator.generate_all(
        generator_linspace=test_generator_linspace,
        stable_parameters=False,
        single_schedule=False,
    )
    print("------------------")
    for time_series in time_series_list:
        print(time_series.samples)
        print(time_series.get_values())
        print("------------------")
    show_plot(time_series_array)
