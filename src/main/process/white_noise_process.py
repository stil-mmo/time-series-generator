import numpy as np
from numpy.typing import NDArray
from src.main.generator_linspace import GeneratorLinspace
from src.main.process.base_parameters_generator import BaseParametersGenerator
from src.main.process.process import Process
from src.main.source_data_processing.aggregated_data import AggregatedData
from src.main.time_series import TimeSeries
from src.main.utils.utils import draw_process_plot


class WNParametersGenerator(BaseParametersGenerator):
    def __init__(
        self,
        lag: int,
        generator_linspace: GeneratorLinspace,
        aggregated_data: AggregatedData | None = None,
    ):
        super().__init__(
            lag=lag,
            generator_linspace=generator_linspace,
            aggregated_data=aggregated_data,
        )

    def generate_parameters(self) -> tuple[float, ...]:
        if self.aggregated_data is None:
            mean = self.generator_linspace.generate_values(is_normal=False)[0]
            std = self.generator_linspace.generate_std()
        else:
            mean = self.aggregated_data.mean_value
            std = self.generator_linspace.generate_std(
                source_value=self.aggregated_data.fraction
            )
        return mean, std

    def generate_init_values(self) -> NDArray:
        return np.array([])


class WhiteNoiseProcess(Process):
    def __init__(
        self,
        generator_linspace: GeneratorLinspace,
        lag: int = 0,
        aggregated_data: AggregatedData | None = None,
    ):
        parameters_generator = WNParametersGenerator(
            lag, generator_linspace, aggregated_data
        )
        super().__init__(
            lag=lag,
            generator_linspace=generator_linspace,
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
            wn_values = np.random.normal(size=data[0], *data[1])
        else:
            std = self.generator_linspace.generate_std()
            wn_values = np.random.normal(previous_values[-1], std, data[0])
        wn_time_series = TimeSeries(data[0])
        wn_time_series.add_values(wn_values, (self.name, data))
        return wn_time_series, self.get_info(data)


if __name__ == "__main__":
    test_generator_linspace = GeneratorLinspace(0.0, 100.0, 100)
    white_noise_process = WhiteNoiseProcess(test_generator_linspace)
    time_series, info = white_noise_process.generate_time_series(
        (100, white_noise_process.parameters_generator.generate_parameters())
    )
    draw_process_plot(time_series, info)
