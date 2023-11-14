from unittest import TestCase, main

from numpy import array
from numpy.testing import assert_array_equal

from src.main.time_series import TimeSeries

EMPTY_SAMPLE = ("", (0, ()))


class TestTimeSeries(TestCase):
    def test_add_values(self):
        ts = TimeSeries()
        ts.add_values(array([1.0, 2.0, 3.0, 4.0]), EMPTY_SAMPLE)
        self.assertEqual(
            assert_array_equal(ts.get_values(), array([1.0, 2.0, 3.0, 4.0])), None
        )

    def test_get_values(self):
        ts = TimeSeries()
        ts.add_values(array([1.0, 2.0, 3.0, 4.0]), EMPTY_SAMPLE)
        self.assertEqual(
            assert_array_equal(ts.get_values(), array([1.0, 2.0, 3.0, 4.0])), None
        )
        self.assertEqual(
            assert_array_equal(ts.get_values(0, 3), array([1.0, 2.0, 3.0])), None
        )


if __name__ == "__main__":
    main()
