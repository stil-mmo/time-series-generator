import pytest

from tsg.linspace_info import LinspaceInfo
from tsg.time_series_generator import TimeSeriesGenerator

GENERATOR_LINSPACE = LinspaceInfo(0.0, 1.0, 100)


def test_generate_all():
    ts_generator = TimeSeriesGenerator(3, 10, GENERATOR_LINSPACE)
    ts_array, ts_list = ts_generator.generate_all()
    assert len(ts_list) == 3
    assert ts_array.shape == (3, 10)


if __name__ == "__main__":
    pytest.main()
