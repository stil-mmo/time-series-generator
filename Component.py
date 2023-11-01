from utils import show_plot

from numpy.random import normal
from numpy import array, ndarray


class Component:
    """
        This class provides basic interface
        for computing various time series components (error, trend, seasonality etc.)

        Attributes
        ----------
        lag : int
            difference between the indexes of the current value
            and the value that participates in the calculation of the current
            (for trend: b_t = b_(t-1) + b*e_t, lag=1,
             for seasonality: s_t = s_(t-12) + c*e_t, lag=12);
        init_values : array(size=lag)
            list of values needed for calculation;
        parameter : float
            coefficient of error;
        error : array
            list of error component values;
        other_values : ndarray(shape=(arrays_count, num_samples))
            required if calculation of component depends on other components;

        Methods
        -------
        set_values(other_values=None)
            computes component values
    """
    def __init__(self, lag, init_values, parameter, error, other_values=None):
        self.num_samples = len(error)
        self.lag = lag
        self.init_values = init_values
        self.parameter = parameter
        self.error = error
        self.values = self.set_values(other_values)

    def set_values(self, other_values=None):
        values = array([0.0 for _ in range(self.num_samples)])
        for i in range(self.lag):
            values[i] = self.init_values[i]
        for i in range(self.num_samples - self.lag):
            value_i = self.parameter * self.error[i]
            if self.lag != 0:
                value_i += values[i]
            if other_values is not None:
                for arr in other_values:
                    value_i += arr[i]
            values[self.lag + i] = value_i
        return values


if __name__ == "__main__":
    s = Component(
        lag=3,
        init_values=array([0.1, 0.5, 0.2]),
        parameter=0.1,
        error=normal(0, 2.5, 10)
    )
    print(s.values)
    other_values = ndarray(shape=(1, 10))
    other_values[0] = s.values
    l = Component(
        lag=1,
        init_values=array([0.0]),
        parameter=0.001,
        error=normal(0, 2.5, 10),
        other_values=other_values
    )
    print(l.values)
    show_plot([s.values, l.values])
