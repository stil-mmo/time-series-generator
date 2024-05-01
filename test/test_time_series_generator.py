import os

import numpy as np
import pytest
from hydra import compose, initialize

from tsg.linspace_info import LinspaceInfo
from tsg.parameters_generation.random_method import RandomMethod
from tsg.process.process_storage import ProcessStorage
from tsg.time_series_generator import TimeSeriesGenerator

GENERATOR_LINSPACE = LinspaceInfo(0.0, 100.0, 100)


def test_generate_all():
    with initialize(version_base="1.2", config_path=os.path.join("..", "config")):
        cfg = compose(config_name="config")
        process_list = ProcessStorage(
            process_list=cfg.scheduler.process_list,
            cfg_process=cfg.process,
            linspace_info=GENERATOR_LINSPACE,
            generation_method=RandomMethod(GENERATOR_LINSPACE),
        )
        ts_generator = TimeSeriesGenerator(
            cfg=cfg, linspace_info=GENERATOR_LINSPACE, process_storage=process_list
        )
        ts_array, ts_list = ts_generator.generate_all()
        assert len(ts_list) == cfg.generation.ts_number
        assert ts_array.shape == (cfg.generation.ts_number, cfg.generation.ts_size)


if __name__ == "__main__":
    pytest.main()
