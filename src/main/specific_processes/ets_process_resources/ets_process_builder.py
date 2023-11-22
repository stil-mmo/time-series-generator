from numpy import array, vstack, zeros
from numpy.random import normal, triangular, uniform
from numpy.typing import NDArray

from src.main.specific_processes.ets_process_resources.ets_component import \
    ETSComponent
from src.main.utils.utils import show_plot

NO_LAG = 0
STABLE_PARAMETER = 1.0


class ETSProcessBuilder:
    def __init__(self, samples_count: int):
        self.num_samples = samples_count
        self.components = zeros(shape=(1, samples_count))
        self.set_normal_error()

    def remove_component(self, index: int) -> None:
        if index < self.components.shape[0]:
            self.components[index] = array([0.0 for _ in range(self.num_samples)])
        else:
            raise ValueError("There is no component with this index")

    def set_normal_error(self, mean=0.0, std=1.0) -> None:
        error = ETSComponent(
            lag=NO_LAG,
            init_values=array([]),
            parameter=STABLE_PARAMETER,
            error=normal(mean, std, self.num_samples),
        )
        self.components[0] = error.values

    def set_uniform_error(self, left=-1.0, right=1.0) -> None:
        error = ETSComponent(
            lag=NO_LAG,
            init_values=array([]),
            parameter=STABLE_PARAMETER,
            error=uniform(left, right, self.num_samples),
        )
        self.components[0] = error.values

    def set_triangular_error(self, left=-1.0, right=1.0, mode=0.0) -> None:
        error = ETSComponent(
            lag=NO_LAG,
            init_values=array([]),
            parameter=STABLE_PARAMETER,
            error=triangular(left, mode, right, self.num_samples),
        )
        self.components[0] = error.values

    def set_long_term(
        self,
        init_value: float,
        parameter: float,
        add_component_indexes: list[int] | None = None,
    ) -> int:
        long_term = ETSComponent(
            lag=1,
            init_values=array([init_value]),
            parameter=float(parameter),
            error=self.components[0],
            additional_values=self.components[add_component_indexes, :]
            if add_component_indexes is not None
            else None,
        )
        self.components = vstack([self.components, long_term.values])
        return self.components.shape[0] - 1

    def set_trend(self, init_value: float, parameter: float) -> int:
        trend = ETSComponent(
            lag=1,
            init_values=array([init_value]),
            parameter=float(parameter),
            error=self.components[0],
        )
        self.components = vstack([self.components, trend.values])
        return self.components.shape[0] - 1

    def set_seasonal(self, lag: int, init_values: NDArray, parameter: float) -> int:
        seasonal = ETSComponent(
            lag=lag,
            init_values=init_values,
            parameter=float(parameter),
            error=self.components[0],
        )
        self.components = vstack([self.components, seasonal.values])
        return self.components.shape[0] - 1

    def generate_values(self) -> NDArray:
        return self.components.sum(axis=0)


def exponential_smoothing(
    num_samples, long_term_init_value, long_term_param, mean, std
):
    ts = ETSProcessBuilder(num_samples)
    ts.set_normal_error(mean=mean, std=std)
    ts.set_long_term(init_value=long_term_init_value, parameter=long_term_param)
    return ts.generate_values()


def trend_model(
    num_samples,
    long_term_init_value,
    trend_init_value,
    long_term_param,
    trend_param,
    mean,
    std,
):
    ts = ETSProcessBuilder(num_samples)
    ts.set_normal_error(mean=mean, std=std)
    trend_index = ts.set_trend(init_value=trend_init_value, parameter=trend_param)
    ts.set_long_term(
        init_value=long_term_init_value,
        parameter=long_term_param,
        add_component_indexes=[trend_index],
    )
    return ts.generate_values()


if __name__ == "__main__":
    show_plot(
        [
            exponential_smoothing(100, 1, 0.05, 0, 1),
            trend_model(100, 1, 0, 0.05, 0.01, 0, 1),
        ]
    )
