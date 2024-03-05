import pytest
from numpy import array
from numpy.testing import assert_array_equal

from tsg.time_series import TimeSeries

EMPTY_SAMPLE = ("", (0, array([])))


def test_add_values():
    ts = TimeSeries(4)
    ts.add_values(array([1.0, 2.0, 3.0, 4.0]), EMPTY_SAMPLE)
    assert_array_equal(ts.get_values(), array([1.0, 2.0, 3.0, 4.0]))


def test_get_values():
    ts = TimeSeries(4)
    ts.add_values(array([1.0, 2.0, 3.0, 4.0]), EMPTY_SAMPLE)
    assert_array_equal(ts.get_values(), array([1.0, 2.0, 3.0, 4.0]))
    assert_array_equal(ts.get_values(0, 3), array([1.0, 2.0, 3.0]))


if __name__ == "__main__":
    pytest.main()
