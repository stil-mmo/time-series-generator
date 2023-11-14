from numpy import ndarray
from scheduler import Scheduler

from src.main.process_list import ProcessList
from src.main.time_series import TimeSeries
from src.main.utils.utils import show_plot


class TimeSeriesGenerator:
    def __init__(
        self,
        num_time_series: int,
        num_steps: int,
        process_list=None,
        process_order=None,
    ):
        self.num_time_series = num_time_series
        self.num_steps = num_steps
        self.process_list = process_list
        self.process_order = process_order

    @staticmethod
    def generate_time_series(
        process_list: ProcessList,
        schedule: list[tuple[str, list[tuple[int, tuple[float, ...]]]]],
        border_values: tuple[float, float],
    ) -> TimeSeries:
        time_series = TimeSeries()
        for process_name, process_schedule in schedule:
            process = process_list.get_processes([process_name])[0]
            for sample in process_schedule:
                if len(time_series.get_values()) < process.lag:
                    time_series.add_values(
                        process.generate_time_series(
                            sample, border_values=border_values
                        )[0].get_values(),
                        (process.name, sample),
                    )
                else:
                    time_series.add_values(
                        process.generate_time_series(
                            sample, previous_values=time_series.get_values()
                        )[0].get_values(),
                        (process.name, sample),
                    )
        return time_series

    def generate_all(
        self,
        border_values: tuple[float, float] = (-10, 10),
        stable_parameters=True,
        single_schedule=True,
    ):
        time_series_array = ndarray((self.num_time_series, self.num_steps))
        scheduler = Scheduler(self.num_steps, self.process_list, self.process_order)
        schedule = (
            None
            if not single_schedule
            else scheduler.generate_schedule(
                border_values[0], border_values[1], stable_parameters
            )
        )
        for i in range(self.num_time_series):
            if schedule is None:
                time_series_array[i] = self.generate_time_series(
                    scheduler.process_list,
                    scheduler.generate_schedule(
                        border_values[0], border_values[1], stable_parameters
                    ),
                    border_values,
                ).get_values()
            else:
                time_series_array[i] = self.generate_time_series(
                    scheduler.process_list, schedule, border_values
                ).get_values()
        return time_series_array


if __name__ == "__main__":
    ts_generator = TimeSeriesGenerator(2, 100)
    show_plot(ts_generator.generate_all())
