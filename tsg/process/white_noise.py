import numpy as np
from numpy.typing import NDArray

from tsg.linspace_info import LinspaceInfo
from tsg.process.process import ParametersGenerator, Process
from tsg.sampling.aggregated_data import AggregatedData
from tsg.time_series import TimeSeries
from tsg.utils.utils import draw_process_plot


class WNParametersGenerator(ParametersGenerator):
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
            mean = self.linspace_info.generate_values(is_normal=False)[0]
            std = self.linspace_info.generate_std()
        else:
            mean = self.aggregated_data.mean_value
            std = self.linspace_info.generate_std(
                source_value=self.aggregated_data.fraction
            )
        return mean, std

    def generate_init_values(self) -> NDArray:
        return np.array([])


class WhiteNoise(Process):
    def __init__(
        self,
        linspace_info: LinspaceInfo,
        lag: int = 0,
        aggregated_data: AggregatedData | None = None,
    ):
        parameters_generator = WNParametersGenerator(
            lag, linspace_info, aggregated_data
        )
        super().__init__(
            lag=lag,
            linspace_info=linspace_info,
            parameters_generator=parameters_generator,
            aggregated_data=aggregated_data,
        )

    @property
    def name(self) -> str:
        return "white_noise"

    @property
    def num_parameters(self) -> int:
        return 2

    def generate_time_series(
        self,
        data: tuple[int, tuple[float, ...]],
        previous_values: NDArray | None = None,
    ) -> tuple[TimeSeries, dict]:
        if previous_values is None:
            wn_values = np.random.normal(data[1][0], data[1][1], size=data[0])
        else:
            std = self.linspace_info.generate_std()
            wn_values = np.random.normal(previous_values[-1], std, data[0])
        wn_time_series = TimeSeries(data[0])
        wn_time_series.add_values(wn_values, (self.name, data))
        return wn_time_series, self.get_info(data)


if __name__ == "__main__":
    test_generator_linspace = LinspaceInfo(0.0, 100.0, 100)
    white_noise_process = WhiteNoise(test_generator_linspace)
    time_series, info = white_noise_process.generate_time_series(
        (100, white_noise_process.parameters_generator.generate_parameters())
    )
    draw_process_plot(time_series, info)
