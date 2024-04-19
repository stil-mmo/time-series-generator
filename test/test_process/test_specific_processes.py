import os.path

import hydra
import numpy as np
import pytest
from numpy import isclose

from tsg.linspace_info import LinspaceInfo
from tsg.parameters_generation.random_method import RandomMethod
from tsg.process.process_storage import ProcessStorage
from tsg.process.simple_random_walk import SimpleRandomWalk
from tsg.time_series import TimeSeries

GENERATOR_LINSPACE = LinspaceInfo(np.float64(0.0), np.float64(100.0), 100)


@pytest.fixture
def process_list():
    with hydra.initialize(version_base="1.2", config_path=os.path.join("..", "..")):
        cfg = hydra.compose(config_name="config")
        process_list = ProcessStorage(cfg=cfg, linspace_info=GENERATOR_LINSPACE)
        return process_list


def test_generate_ts(process_list):
    for process in process_list.processes.values():
        data = (100, process.parameters_generator.generate_parameters())
        time_series, info = process.generate_time_series(data)
        assert isinstance(time_series, TimeSeries)
        assert isinstance(info, dict)
        assert len(time_series.values) == 100


def test_generate_ts_with_values(process_list):
    for process in process_list.processes.values():
        previous_values = np.random.uniform(0, 1, 10)
        data = (100, process.parameters_generator.generate_parameters())
        time_series, info = process.generate_time_series(data, previous_values)
        assert isinstance(time_series, TimeSeries)
        assert isinstance(info, dict)
        assert len(time_series.values) == 100


def test_random_walk():
    with hydra.initialize(version_base="1.2", config_path=os.path.join("..", "..")):
        cfg = hydra.compose(config_name="config")
        generation_method = RandomMethod(
            parameters_generation_cfg=cfg.parameters_generation_method,
            linspace_info=GENERATOR_LINSPACE,
        )
        simple_random_walk = SimpleRandomWalk(
            GENERATOR_LINSPACE, generation_method, fixed_walk=1.0
        )
        data = (100, simple_random_walk.parameters_generator.generate_parameters())
        time_series, info = simple_random_walk.generate_time_series(data)

        # Check if the generated time series follows the random walk rule
        values = time_series.get_values()
        for i in range(1, len(values)):
            diff = abs(values[i] - values[i - 1])
            assert isclose(diff, 1.0)  # Assuming fixed_walk is set to 1.0


if __name__ == "__main__":
    pytest.main()
