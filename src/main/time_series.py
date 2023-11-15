"""This module contains the TimeSeries class"""

from numpy import array, concatenate


class TimeSeries:
    """This class provides methods to store time series values and metadata"""

    def __init__(self):
        self.values = array([])
        self.samples = []

    def add_values(
        self, new_values: array, new_sample: tuple[str, tuple[int, tuple]]
    ) -> None:
        """Adds new values to the time series"""
        if len(self.values) == 0:
            self.values = new_values
            self.samples.append(new_sample)
        else:
            self.values = concatenate((self.values, new_values))
            self.samples.append(new_sample)

    def get_values(self, start_index=0, end_index=None) -> array:
        """Returns time series values"""
        return self.values[start_index:end_index]
