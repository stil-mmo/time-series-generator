import numpy as np
from numpy.typing import NDArray
from src.main.generator_linspace import GeneratorLinspace
from src.main.process.base_parameters_generator import BaseParametersGenerator
from src.main.process.ets_process_resources.ets_process_builder import ETSProcessBuilder
from src.main.process.process import Process
from src.main.source_data_processing.aggregated_data import AggregatedData
from src.main.time_series import TimeSeries
from src.main.utils.utils import draw_process_plot


class TESParametersGenerator(BaseParametersGenerator):
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
            std = self.generator_linspace.generate_std()
            long_term_coefficient = np.random.uniform(0.0, 0.3)
            trend_coefficient = np.random.uniform(0.0, 0.05)
            seasonality_coefficient = np.random.uniform(0.5, 1.0)
        else:
            std = self.generator_linspace.generate_std(
                source_value=self.aggregated_data.fraction
            )
            long_term_coefficient = self.aggregated_data.fraction / 3
            trend_coefficient = long_term_coefficient / 20.0
            seasonality_coefficient = 1 - (self.aggregated_data.fraction / 2)
        return long_term_coefficient, trend_coefficient, seasonality_coefficient, std

    def generate_init_values(self) -> NDArray:
        if self.aggregated_data is None:
            init_values = np.zeros((3, self.lag))
            init_values[0][0] = self.generator_linspace.generate_values()
            init_values[1][0] = 0.0
            init_values[2] = self.generator_linspace.generate_values(
                num_values=self.lag
            )
        else:
            init_values = np.zeros((3, self.lag))
            init_values[0][0] = self.aggregated_data.mean_value
            init_values[1][0] = 0.0
            init_values[2][0] = 0.0
            for i in range(1, self.lag):
                init_values[2][i] = init_values[2][i - 1] + np.random.normal(
                    0.0, self.generator_linspace.step / 2
                )
        return init_values


class TripleExponentialSmoothing(Process):
    def __init__(
        self,
        generator_linspace: GeneratorLinspace,
        lag: int,
        aggregated_data: AggregatedData | None = None,
    ):
        parameters_generator = TESParametersGenerator(
            lag=lag,
            generator_linspace=generator_linspace,
            aggregated_data=aggregated_data,
        )
        super().__init__(
            lag=lag,
            generator_linspace=generator_linspace,
            parameters_generator=parameters_generator,
            aggregated_data=aggregated_data,
        )

    @property
    def name(self) -> str:
        return "triple_exponential_smoothing"

    @property
    def num_parameters(self) -> int:
        return 4

    def generate_time_series(
        self,
        data: tuple[int, tuple[float, ...]],
        previous_values: NDArray | None = None,
    ) -> tuple[TimeSeries, dict]:
        ets_values = ETSProcessBuilder(data[0])
        ets_values.set_normal_error(mean=0.0, std=data[1][3])
        if previous_values is None or len(previous_values) < 1:
            init_values = self.parameters_generator.generate_init_values()
            long_term_init_value = init_values[0][0]
            trend_init_value = init_values[1][0]
            seasonality_init_values = init_values[2]
        elif len(previous_values) < self.lag:
            init_values = self.parameters_generator.generate_init_values()
            long_term_init_value = previous_values[-1]
            trend_init_value = init_values[1][0]
            seasonality_init_values = np.array([0.0 for _ in range(self.lag)])
            seasonality_init_values[: len(previous_values)] = previous_values
            for i in range(len(previous_values), self.lag):
                seasonality_init_values[i] = np.random.normal(
                    previous_values[-1], self.generator_linspace.step
                )
        else:
            long_term_init_value = previous_values[-1]
            trend_init_value = 0.0
            seasonality_init_values = previous_values[-self.lag :]
            seasonality_init_values /= np.sum(seasonality_init_values)
        ets_values.set_seasonal(
            lag=self.lag,
            init_values=seasonality_init_values,
            parameter=data[1][2],
        )
        trend_index = ets_values.set_trend(
            init_value=trend_init_value, parameter=data[1][1]
        )
        ets_values.set_long_term(
            init_value=long_term_init_value,
            parameter=data[1][0],
            add_component_indexes=[trend_index],
        )
        trend_time_series = TimeSeries(data[0])
        trend_time_series.add_values(ets_values.generate_values(), (self.name, data))
        return trend_time_series, self.get_info(
            data, (long_term_init_value, trend_init_value)
        )


if __name__ == "__main__":
    test_generator_linspace = GeneratorLinspace(0.0, 100.0, 100)
    proc = TripleExponentialSmoothing(test_generator_linspace, 12)
    test_sample = (100, proc.parameters_generator.generate_parameters())
    test_time_series, test_info = proc.generate_time_series(test_sample)
    draw_process_plot(test_time_series, test_info)
