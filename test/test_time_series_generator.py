import numpy as np
import pytest
from hydra import compose, initialize

from tsg.linspace_info import LinspaceInfo
from tsg.time_series_generator import TimeSeriesGenerator

GENERATOR_LINSPACE = LinspaceInfo(0.0, 100.0, 100)


def test_generate_all():
    with initialize(version_base=None, config_path=".."):
        cfg = compose(config_name="config")
        ts_generator = TimeSeriesGenerator(cfg, GENERATOR_LINSPACE)
        ts_array, ts_list = ts_generator.generate_all()
        assert len(ts_list) == cfg.generation.ts_number
        assert ts_array.shape == (cfg.generation.ts_number, cfg.generation.ts_size)


if __name__ == "__main__":
    pytest.main()
