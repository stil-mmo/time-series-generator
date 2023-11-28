from random import uniform

from numpy import array, sum
from numpy.random import normal
from numpy.typing import NDArray
from src.main.generator_linspace import GeneratorLinspace
from src.main.process import Process
from src.main.time_series import TimeSeries
from src.main.utils.parameters_approximation import weighted_mean
from src.main.utils.utils import draw_process_plot


class SimpleRandomWalk(Process):
    def __init__(
        self,
        generator_linspace: GeneratorLinspace,
        lag: int = 1,
        fixed_walk: float | None = None,
    ):
        super().__init__(lag, generator_linspace)
        self.fixed_walk = fixed_walk

    @property
    def name(self) -> str:
        return "simple_random_walk"

    @property
    def num_parameters(self) -> int:
        return 3

    def calculate_data(
        self, source_values: NDArray | None = None
    ) -> tuple[tuple[float, ...], NDArray]:
        if source_values is None:
            return self.generate_parameters(), self.generate_init_values()
        mean = weighted_mean(source_values)
        up_probability = mean / sum(source_values)
        down_probability = 1 - up_probability
        step = self.generator_linspace.calculate_std(mean)
        return (up_probability, down_probability, step), array([mean / 2])

    def generate_parameters(self) -> tuple[float, ...]:
        up_probability = uniform(0.0, 1.0)
        down_probability = 1 - up_probability
        step = self.generator_linspace.step
        walk = normal(step, step / 2) if self.fixed_walk is None else self.fixed_walk
        return up_probability, down_probability, walk

    def generate_init_values(self) -> NDArray:
        return self.generator_linspace.generate_values(is_normal=False)

    def generate_time_series(
        self,
        data: tuple[int, tuple[float, ...]],
        previous_values: NDArray | None = None,
    ) -> tuple[TimeSeries, dict]:
        up_probability, down_probability, walk = data[1]
        values = array([0.0 for _ in range(0, data[0])])
        values_to_add = data[0]
        if previous_values is None:
            init_values = self.generate_init_values()
            values[0 : len(init_values)] = init_values
            values_to_add -= len(init_values)
            previous_value = init_values[-1]
        else:
            previous_value = previous_values[-1]
        for i in range(data[0] - values_to_add, data[0]):
            if uniform(0, 1) < up_probability:
                previous_value += walk
            else:
                previous_value -= walk
            values[i] = previous_value
        rw_time_series = TimeSeries(data[0])
        rw_time_series.add_values(values, (self.name, data))
        if previous_values is None:
            return rw_time_series, self.get_info(
                data, values[0 : data[0] - values_to_add]
            )
        else:
            return rw_time_series, self.get_info(data, (previous_values[-1],))


if __name__ == "__main__":
    test_generator_linspace = GeneratorLinspace(0.0, 100.0, 100)
    proc = SimpleRandomWalk(test_generator_linspace)
    test_sample = (100, proc.generate_parameters())
    test_time_series, test_info = proc.generate_time_series(test_sample)
    print(test_time_series.get_values())
    draw_process_plot(test_time_series, test_info)
