import numpy as np
from numpy.typing import NDArray

from tsg.linspace_info import LinspaceInfo
from tsg.process.process import ParametersGenerator, Process
from tsg.sampling.aggregated_data import AggregatedData
from tsg.time_series import TimeSeries
from tsg.utils.utils import draw_process_plot


class SRWParametersGenerator(ParametersGenerator):
    def __init__(
        self,
        lag: int,
        linspace_info: LinspaceInfo,
        aggregated_data: AggregatedData | None = None,
        fixed_walk: float | None = None,
    ):
        super().__init__(
            lag=lag,
            linspace_info=linspace_info,
            aggregated_data=aggregated_data,
        )
        self.fixed_walk = fixed_walk

    def generate_parameters(self) -> NDArray[np.float64]:
        if self.aggregated_data is None:
            up_probability = np.random.uniform(0.0, 1.0)
            down_probability = 1 - up_probability
            walk = (
                self.linspace_info.generate_std()
                if self.fixed_walk is None
                else self.fixed_walk
            )
        else:
            up_probability = self.aggregated_data.fraction
            down_probability = 1 - up_probability
            walk = self.linspace_info.generate_std(
                source_value=self.aggregated_data.fraction
            )
        return np.array([up_probability, down_probability, walk])

    def generate_init_values(self) -> NDArray[np.float64]:
        if self.aggregated_data is None:
            values = self.linspace_info.generate_values(is_normal=False)
        else:
            values = np.array([self.aggregated_data.mean_value / 2])
        return values


class SimpleRandomWalk(Process):
    def __init__(
        self,
        linspace_info: LinspaceInfo,
        lag: int = 1,
        aggregated_data: AggregatedData | None = None,
        fixed_walk: float | None = None,
    ):
        parameters_generator = SRWParametersGenerator(
            lag=lag,
            linspace_info=linspace_info,
            aggregated_data=aggregated_data,
            fixed_walk=fixed_walk,
        )
        super().__init__(
            lag=lag,
            linspace_info=linspace_info,
            parameters_generator=parameters_generator,
            aggregated_data=aggregated_data,
        )

    @property
    def name(self) -> str:
        return "simple_random_walk"

    @property
    def num_parameters(self) -> int:
        return 3

    def generate_time_series(
        self,
        data: tuple[int, NDArray[np.float64]],
        previous_values: NDArray[np.float64] | None = None,
    ) -> tuple[TimeSeries, dict]:
        up_probability, down_probability, walk = data[1]
        values = np.array([0.0 for _ in range(0, data[0])])
        values_to_add = data[0]
        if previous_values is None:
            init_values = self.parameters_generator.generate_init_values()
            values[0 : len(init_values)] = init_values
            values_to_add -= len(init_values)
            previous_value = init_values[-1]
        else:
            previous_value = previous_values[-1]
        for i in range(data[0] - values_to_add, data[0]):
            if np.random.uniform(0, 1) < up_probability:
                previous_value += walk
            else:
                previous_value -= walk
            values[i] = previous_value
        rw_time_series = TimeSeries(data[0])
        rw_time_series.add_values(values, (self.name, data))
        if previous_values is None:
            return rw_time_series, self.get_info(
                data, np.array(values[0 : data[0] - values_to_add])
            )
        else:
            return rw_time_series, self.get_info(data, np.array([previous_values[-1]]))


if __name__ == "__main__":
    test_generator_linspace = LinspaceInfo(np.float64(0.0), np.float64(100.0), 100)
    proc = SimpleRandomWalk(test_generator_linspace)
    test_sample = (100, proc.parameters_generator.generate_parameters())
    test_time_series, test_info = proc.generate_time_series(test_sample)
    print(test_time_series.get_values())
    draw_process_plot(test_time_series, test_info)
