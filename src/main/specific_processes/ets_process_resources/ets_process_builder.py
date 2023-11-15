"""This module contains the ETSProcessBuilder class, which is used to build ETS processes"""

from ets_component import ETSComponent
from numpy import array, vstack, zeros
from numpy.random import normal, triangular, uniform

NO_LAG = 0
STABLE_PARAMETER = 1.0


class ETSProcessBuilder:
    """Provides methods to build ETS processes"""

    def __init__(self, samples_count):
        self.num_samples = samples_count
        self.components = zeros(shape=(1, samples_count))
        self.set_normal_error()

    def remove_component(self, index):
        """Removes component with specified index"""
        if index < self.components.shape[0]:
            self.components[index] = array([0.0 for _ in range(self.num_samples)])
        else:
            raise ValueError("There is no component with this index")

    def set_normal_error(self, mean=0.0, std=1.0):
        """Sets normal error component with specified mean and standard deviation"""
        error = ETSComponent(
            lag=NO_LAG,
            init_values=array([]),
            parameter=STABLE_PARAMETER,
            error=normal(mean, std, self.num_samples),
        )
        self.components[0] = error.values

    def set_uniform_error(self, left=-1.0, right=1.0):
        """Sets uniform error component with specified left and right bounds"""
        error = ETSComponent(
            lag=NO_LAG,
            init_values=array([]),
            parameter=STABLE_PARAMETER,
            error=uniform(left, right, self.num_samples),
        )
        self.components[0] = error.values

    def set_triangular_error(self, left=-1.0, right=1.0, mode=0.0):
        """Sets triangular error component with specified left, right bounds and mode"""
        error = ETSComponent(
            lag=NO_LAG,
            init_values=array([]),
            parameter=STABLE_PARAMETER,
            error=triangular(left, mode, right, self.num_samples),
        )
        self.components[0] = error.values

    def set_long_term(self, init_value, parameter, add_component_indexes=None):
        """Sets long term component with specified initial value and parameter"""
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

    def set_trend(self, init_value, parameter):
        """Sets trend component with specified initial value and parameter"""
        trend = ETSComponent(
            lag=1,
            init_values=array([init_value]),
            parameter=float(parameter),
            error=self.components[0],
        )
        self.components = vstack([self.components, trend.values])
        return self.components.shape[0] - 1

    def set_seasonal(self, lag, init_values, parameter):
        """Sets seasonal component with specified lag, initial values and parameter"""
        seasonal = ETSComponent(
            lag=lag,
            init_values=init_values,
            parameter=float(parameter),
            error=self.components[0],
        )
        self.components = vstack([self.components, seasonal.values])
        return self.components.shape[0] - 1

    def generate_time_series(self):
        """Generates time series based on components"""
        return self.components.sum(axis=0)
