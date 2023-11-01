from Component import Component
from utils import show_plot

from numpy.random import normal, uniform, triangular
from numpy import array, zeros


class TimeSeriesGenerator:
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.components = zeros(shape=(4, num_samples))
        self.set_normal_error()

    def remove_component(self, index):
        self.components[index] = array([0. for _ in range(self.num_samples)])

    def set_normal_error(self, mean=0.0, std=1.0):
        error = Component(
            lag=0, init_values=array([]),
            parameter=1., error=normal(mean, std, self.num_samples)
        )
        self.components[0] = error.values

    def set_uniform_error(self, left=-1.0, right=-1.0):
        error = Component(
            lag=0, init_values=array([]),
            parameter=1., error=uniform(left, right, self.num_samples)
        )
        self.components[0] = error.values

    def set_triangular_error(self, left=-1.0, right=-1.0, mode=0.0):
        error = Component(
            lag=0, init_values=array([]),
            parameter=1., error=triangular(left, mode, right, self.num_samples)
        )
        self.components[0] = error.values

    def set_long_term(self, l_0, a, indexes=None):
        long_term = Component(
            lag=1, init_values=array([l_0]),
            parameter=float(a), error=self.components[0],
            other_values=self.components[indexes, :] if indexes is not None else None)
        self.components[1] = long_term.values

    def set_trend(self, b_0, b):
        trend = Component(
            lag=1, init_values=array([b_0]),
            parameter=float(b), error=self.components[0])
        self.components[2] = trend.values

    def set_seasonal(self, lag, init_values, c):
        seasonal = Component(
            lag=lag, init_values=init_values,
            parameter=float(c), error=self.components[0])
        self.components[3] = seasonal.values

    def generate_time_series(self):
        return self.components.sum(axis=0)


def ANN(num_samples, l_0, a, mean, std):
    ts = TimeSeriesGenerator(num_samples)
    ts.set_normal_error(mean=mean, std=std)
    ts.set_long_term(l_0=l_0, a=a)
    return ts.generate_time_series()


def AAN(num_samples, l_0, b_0, a, b, mean, std):
    ts = TimeSeriesGenerator(num_samples)
    ts.set_normal_error(mean=mean, std=std)
    ts.set_trend(b_0=b_0, b=b)
    ts.set_long_term(l_0=l_0, a=a, indexes=[2])
    return ts.generate_time_series()


if __name__ == "__main__":
    show_plot([ANN(100, 1, 0.05, 0, 1), AAN(100, 1, 0, 0.05, 0.01, 0, 1)])
