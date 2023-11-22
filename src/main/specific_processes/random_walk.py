from numpy import array
from numpy.random import normal
from numpy.typing import NDArray
from src.main.generator_linspace import GeneratorLinspace
from src.main.process import Process
from src.main.time_series import TimeSeries
from src.main.utils.utils import draw_process_plot


class RandomWalk(Process):
    def __init__(self, generator_linspace: GeneratorLinspace, lag: int = 1):
        super().__init__(lag, generator_linspace)

    @property
    def name(self) -> str:
        return "random_walk"

    @property
    def num_parameters(self) -> int:
        return 1

    def create_parameters(self, source_values: NDArray) -> tuple[float, ...]:
        return self.generate_parameters()

    def generate_parameters(self) -> tuple[float, ...]:
        return (self.generator_linspace.generate_std(),)

    def generate_init_values(self) -> NDArray:
        return self.generator_linspace.generate_values(is_normal=False)

    def generate_time_series(
        self,
        sample: tuple[int, tuple[float, ...]],
        previous_values: NDArray | None = None,
    ) -> tuple[TimeSeries, dict]:
        values = array([0.0 for _ in range(0, sample[0])])
        values_to_add = sample[0]
        if previous_values is None:
            values[0] = self.generate_init_values()[0]
            previous_value = values[0]
            values_to_add -= 1
        else:
            previous_value = previous_values[-1]
        for i in range(sample[0] - values_to_add, sample[0]):
            previous_value += normal(0.0, sample[1][0])
            values[i] = previous_value
        rw_time_series = TimeSeries(sample[0])
        rw_time_series.add_values(values, (self.name, sample))
        if previous_values is None:
            return rw_time_series, self.get_info(
                sample, values[0 : sample[0] - values_to_add]
            )
        else:
            return rw_time_series, self.get_info(sample, (previous_values[-1],))


if __name__ == "__main__":
    test_generator_linspace = GeneratorLinspace(0.0, 100.0, 100)
    proc = RandomWalk(test_generator_linspace)
    test_sample = (100, proc.generate_parameters())
    test_time_series, test_info = proc.generate_time_series(test_sample)
    print(test_time_series.get_values())
    draw_process_plot(test_time_series, test_info)
