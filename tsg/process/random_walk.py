import numpy as np
from numpy.typing import NDArray

from tsg.linspace_info import LinspaceInfo
from tsg.process.process import ParametersGenerator, Process
from tsg.sampling.aggregated_data import AggregatedData
from tsg.time_series import TimeSeries
from tsg.utils.utils import draw_process_plot


class RWParametersGenerator(ParametersGenerator):
    def __init__(
        self,
        lag: int,
        linspace_info: LinspaceInfo,
        aggregated_data: AggregatedData | None = None,
    ):
        super().__init__(
            lag=lag,
            linspace_info=linspace_info,
            aggregated_data=aggregated_data,
        )

    def generate_parameters(self) -> tuple[float, ...]:
        if self.aggregated_data is None:
            std = self.linspace_info.generate_std()
        else:
            std = self.linspace_info.generate_std(
                source_value=self.aggregated_data.fraction
            )
        return (std,)

    def generate_init_values(self) -> NDArray:
        if self.aggregated_data is None:
            values = self.linspace_info.generate_values(is_normal=False)
        else:
            values = np.array([self.aggregated_data.mean_value / 2])
        return values


class RandomWalk(Process):
    def __init__(
        self,
        linspace_info: LinspaceInfo,
        lag: int = 1,
        aggregated_data: AggregatedData | None = None,
    ):
        parameters_generator = RWParametersGenerator(
            lag=lag,
            linspace_info=linspace_info,
            aggregated_data=aggregated_data,
        )
        super().__init__(
            lag=lag,
            linspace_info=linspace_info,
            parameters_generator=parameters_generator,
            aggregated_data=aggregated_data,
        )

    @property
    def name(self) -> str:
        return "random_walk"

    @property
    def num_parameters(self) -> int:
        return 1

    def generate_time_series(
        self,
        data: tuple[int, tuple[float, ...]],
        previous_values: NDArray | None = None,
    ) -> tuple[TimeSeries, dict]:
        values = np.array([0.0 for _ in range(0, data[0])])
        values_to_add = data[0]
        if previous_values is None:
            values[0] = self.parameters_generator.generate_init_values()[0]
            previous_value = values[0]
            values_to_add -= 1
        else:
            previous_value = previous_values[-1]
        for i in range(data[0] - values_to_add, data[0]):
            previous_value += np.random.normal(0.0, data[1][0])
            values[i] = previous_value
        rw_time_series = TimeSeries(data[0])
        rw_time_series.add_values(values, (self.name, data))
        if previous_values is None:
            return rw_time_series, self.get_info(
                data, values[0 : data[0] - values_to_add]
            )
        else:
            return rw_time_series, self.get_info(data, np.array([previous_values[-1]]))


if __name__ == "__main__":
    test_generator_linspace = LinspaceInfo(0.0, 100.0, 100)
    proc = RandomWalk(test_generator_linspace)
    test_sample = (100, proc.parameters_generator.generate_parameters())
    test_time_series, test_info = proc.generate_time_series(test_sample)
    print(test_time_series.get_values())
    draw_process_plot(test_time_series, test_info)
