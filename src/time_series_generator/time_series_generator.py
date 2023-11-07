from component import Component
from numpy import array, vstack, zeros
from numpy.random import normal, triangular, uniform

from utils.utils import show_plot

NO_LAG = 0
STABLE_PARAMETER = 1.0


class TimeSeriesGenerator:
    def __init__(self, samples_count):
        self.num_samples = samples_count
        self.components = zeros(shape=(1, samples_count))
        self.set_normal_error()

    def remove_component(self, index):
        if index < self.components.shape[0]:
            self.components[index] = array([0.0 for _ in range(self.num_samples)])
        else:
            raise ValueError("There is no component with this index")

    def set_normal_error(self, mean=0.0, std=1.0):
        error = Component(
            lag=NO_LAG,
            init_values=array([]),
            parameter=STABLE_PARAMETER,
            error=normal(mean, std, self.num_samples),
        )
        self.components[0] = error.values

    def set_uniform_error(self, left=-1.0, right=1.0):
        error = Component(
            lag=NO_LAG,
            init_values=array([]),
            parameter=STABLE_PARAMETER,
            error=uniform(left, right, self.num_samples),
        )
        self.components[0] = error.values

    def set_triangular_error(self, left=-1.0, right=1.0, mode=0.0):
        error = Component(
            lag=NO_LAG,
            init_values=array([]),
            parameter=STABLE_PARAMETER,
            error=triangular(left, mode, right, self.num_samples),
        )
        self.components[0] = error.values

    def set_long_term(self, init_value, parameter, add_component_indexes=None):
        long_term = Component(
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

    def set_trend(self, init_value, parameter):
        trend = Component(
            lag=1,
            init_values=array([init_value]),
            parameter=float(parameter),
            error=self.components[0],
        )
        self.components = vstack([self.components, trend.values])
        return self.components.shape[0] - 1

    def set_seasonal(self, lag, init_values, parameter):
        seasonal = Component(
            lag=lag,
            init_values=init_values,
            parameter=float(parameter),
            error=self.components[0],
        )
        self.components = vstack([self.components, seasonal.values])
        return self.components.shape[0] - 1

    def generate_time_series(self):
        return self.components.sum(axis=0)


def ANN(num_samples, long_term_init_value, long_term_param, mean, std):
    ts = TimeSeriesGenerator(num_samples)
    ts.set_normal_error(mean=mean, std=std)
    ts.set_long_term(init_value=long_term_init_value, parameter=long_term_param)
    return ts.generate_time_series()


def AAN(
    num_samples,
    long_term_init_value,
    trend_init_value,
    long_term_param,
    trend_param,
    mean,
    std,
):
    ts = TimeSeriesGenerator(num_samples)
    ts.set_normal_error(mean=mean, std=std)
    trend_index = ts.set_trend(init_value=trend_init_value, parameter=trend_param)
    ts.set_long_term(
        init_value=long_term_init_value,
        parameter=long_term_param,
        add_component_indexes=[trend_index],
    )
    return ts.generate_time_series()


if __name__ == "__main__":
    show_plot([ANN(100, 1, 0.05, 0, 1), AAN(100, 1, 0, 0.05, 0.01, 0, 1)])
