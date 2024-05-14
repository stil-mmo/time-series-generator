import os.path

import pytest
from hydra import compose, initialize

from tsg.linspace_info import LinspaceInfo
from tsg.parameters_generation.random_method import RandomMethod
from tsg.process.process_storage import ProcessStorage

GENERATOR_LINSPACE = LinspaceInfo(0.0, 100.0, 100)


def test_add_processes():
    with initialize(version_base="1.2", config_path=os.path.join("..", "config")):
        cfg = compose(config_name="config")
        cfg.scheduler.process_list = []
        process_list = ProcessStorage(
            process_list=cfg.scheduler.process_list,
            cfg_process=cfg.process,
            linspace_info=GENERATOR_LINSPACE,
            generation_method=RandomMethod(GENERATOR_LINSPACE),
        )
        process_list.add_processes(["white_noise"])
        assert len(process_list.processes) == 1
        process_list.add_processes(["white_noise"])
        assert process_list.num_processes == 1


def test_get_processes():
    with initialize(version_base="1.2", config_path=os.path.join("..", "config")):
        cfg = compose(config_name="config")
        process_list = ProcessStorage(
            process_list=cfg.scheduler.process_list,
            cfg_process=cfg.process,
            linspace_info=GENERATOR_LINSPACE,
            generation_method=RandomMethod(GENERATOR_LINSPACE),
        )
        assert process_list.get_processes(["white_noise"])[0].name == "white_noise"


def test_get_random_processes():
    with initialize(version_base="1.2", config_path=os.path.join("..", "config")):
        cfg = compose(config_name="config")
        process_list = ProcessStorage(
            process_list=cfg.scheduler.process_list,
            cfg_process=cfg.process,
            linspace_info=GENERATOR_LINSPACE,
            generation_method=RandomMethod(GENERATOR_LINSPACE),
        )
        assert len(process_list.get_random_processes(5)) == 5


if __name__ == "__main__":
    pytest.main()
