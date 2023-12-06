from numpy import array, ndarray, transpose
from numpy.random import normal
from numpy.typing import NDArray

from src.main.utils.utils import show_plot


class ETSComponent:
    """
    This class provides basic interface
    for computing various time series components (error, trend, seasonality etc.)

    Attributes
    ----------
    lag : int
        difference between the add_component_indexes of the current value
        and the value that participates in the calculation of the current
        (for trend: b_t = b_(t-1) + parameter*e_t, lag=1,
         for seasonality: s_t = s_(t-12) + parameter*e_t, lag=12);
    init_values : array(size=lag)
        list of values needed for calculation;
    parameter : float
        coefficient of error;
    error : array
        list of error component values;
    additional_values : ndarray(shape=(arrays_count, samples_count))
        required if calculation of component depends on other components;

    Methods
    -------
    set_values(additional_values=None)
        computes component values
    """

    def __init__(
        self,
        lag: int,
        init_values: NDArray,
        parameter: float,
        error: NDArray,
        additional_values: NDArray | None = None,
    ):
        self.num_samples = len(error)
        self.lag = lag
        self.init_values = init_values
        self.parameter = parameter
        self.error = error
        self.values = self.set_values(additional_values)

    def set_values(self, additional_values=None):
        component_values = array([0.0 for _ in range(self.num_samples + self.lag)])
        for i in range(self.lag):
            component_values[i] = self.init_values[i]
        for i in range(self.num_samples):
            component_value = self.parameter * self.error[i]
            if self.lag != 0:
                component_value += component_values[i]
            if additional_values is not None:
                component_value += transpose(additional_values[:, i])
            component_values[self.lag + i] = component_value
        return component_values[self.lag :]


if __name__ == "__main__":
    seasonal = ETSComponent(
        lag=3,
        init_values=array([0.1, 0.5, 0.2]),
        parameter=0.1,
        error=normal(0, 2.5, 10),
    )
    print(seasonal.values)
    other_values = ndarray(shape=(1, 10))
    other_values[0] = seasonal.values
    long_term = ETSComponent(
        lag=1,
        init_values=array([0.0]),
        parameter=0.001,
        error=normal(0, 2.5, 10),
        additional_values=other_values,
    )
    print(long_term.values)
    show_plot([seasonal.values, long_term.values])
