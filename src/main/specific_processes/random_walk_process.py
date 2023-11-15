"""This module contains the RandomWalkProcess class"""

from random import uniform

from numpy import array

from src.main.process import Process
from src.main.time_series import TimeSeries
from src.main.utils.utils import draw_process_plot


class RandomWalkProcess(Process):
    """Random walk process class"""

    @property
    def name(self) -> str:
        """Process name"""
        return "random_walk"

    @property
    def lag(self) -> int:
        """Process lag (number of previous elements required to compute next element)"""
        return 1

    @property
    def num_parameters(self) -> int:
        """Process parameters number"""
        return 2

    def generate_parameters(
        self, low_value: float = 0.0, high_value: float = 0.0
    ) -> tuple[float, float]:
        """Generates process parameters"""
        # pylint: disable=unused-argument
        up_probability = uniform(0, 1)
        down_probability = 1 - up_probability
        return up_probability, down_probability

    def generate_init_values(self, low_value: float, high_value: float) -> array:
        """Generates process initial values (values number is equal to lag)"""
        return array([uniform(low_value, high_value)])

    def get_info(
        self, sample: tuple[int, tuple], init_values: tuple[float, ...] = None
    ) -> dict:
        """Returns information about process"""
        info = {
            "name": self.name,
            "lag": self.lag,
            "up_probability": sample[1][0],
            "down_probability": sample[1][1],
            "initial_values": init_values,
        }
        return info

    def generate_time_series(
        self,
        sample: tuple[int, tuple],
        previous_values: array = None,
        border_values: tuple[float, float] = None,
    ) -> tuple[TimeSeries, dict]:
        """Generates time series with process"""
        up_probability, _ = sample[1]
        values = array([0.0 for _ in range(0, sample[0])])
        values_to_add = sample[0]
        if previous_values is None:
            init_values = self.generate_init_values(border_values[0], border_values[1])
            values[0 : len(init_values)] = init_values
            values_to_add -= len(init_values)
            previous_value = init_values[-1]
        else:
            previous_value = previous_values[-1]
        for i in range(sample[0] - values_to_add, sample[0]):
            if uniform(0, 1) < up_probability:
                previous_value += 1.0
            else:
                previous_value -= 1.0
            values[i] = previous_value
        rw_time_series = TimeSeries()
        rw_time_series.add_values(values, (self.name, sample))
        if previous_values is None:
            return rw_time_series, self.get_info(
                sample, values[0 : sample[0] - values_to_add]
            )
        return rw_time_series, self.get_info(sample, (previous_values[-1],))

    def draw_plot(
        self,
        border_values: tuple[float, float] = None,
        path: str = None,
        time_series_data: tuple[TimeSeries, dict] = None,
    ):
        """Draws process plot"""
        if time_series_data is None:
            sample = self.generate_parameters(border_values[0], border_values[1])
            init_values = self.generate_init_values(border_values[0], border_values[1])
            data = self.generate_time_series((100, sample), init_values)
        else:
            data = time_series_data
        if path is not None:
            draw_process_plot(
                data[0].get_values(), data[1], path=f"{path}\\{self.name}_plot.png"
            )
        else:
            draw_process_plot(data[0].get_values(), data[1])
