from numpy import ndarray
from scheduler import PROCESS_ORDER, PROCESS_SAMPLES, Scheduler
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
    ):
        self.num_time_series = num_time_series
        self.num_steps = num_steps
        self.process_list = process_list
        self.process_order = process_order

    def generate_time_series(
        self,
        process_list: ProcessList,
        schedule: list[PROCESS_SAMPLES],
    ) -> TimeSeries:
        time_series = TimeSeries(self.num_steps)
        for process_name, process_schedule in schedule:
            process = process_list.get_processes([process_name])[0]
            for sample in process_schedule:
                if len(time_series.get_values()) < process.lag:
                    time_series.add_values(
                        process.generate_time_series(sample)[0].get_values(),
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
        stable_parameters: bool = True,
        single_schedule: bool = True,
    ):
        time_series_array = ndarray((self.num_time_series, self.num_steps))
        scheduler = Scheduler(
            num_steps=self.num_steps,
            border_values=border_values,
            process_list=self.process_list,
            process_order=self.process_order,
        )
        schedule = (
            None
            if not single_schedule
            else scheduler.generate_schedule(stable_parameters=stable_parameters)
        )
        for i in range(self.num_time_series):
            if schedule is None:
                time_series_array[i] = self.generate_time_series(
                    scheduler.process_list,
                    scheduler.generate_schedule(stable_parameters=stable_parameters),
                ).get_values()
            else:
                time_series_array[i] = self.generate_time_series(
                    scheduler.process_list, schedule
                ).get_values()
        return time_series_array


if __name__ == "__main__":
    ts_generator = TimeSeriesGenerator(2, 100)
    show_plot(ts_generator.generate_all())
