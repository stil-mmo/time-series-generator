import numpy as np
import pytest

from tsg.linspace_info import LinspaceInfo
from tsg.process.process_list import ProcessList
from tsg.process.white_noise import WhiteNoise

GENERATOR_LINSPACE = LinspaceInfo(np.float64(0.0), np.float64(1.0), 100)


def test_add_processes():
    process_list = ProcessList()
    process_list.add_processes([WhiteNoise(GENERATOR_LINSPACE)])
    assert len(process_list.processes) == 1
    process_list.add_processes([WhiteNoise(GENERATOR_LINSPACE)])
    assert process_list.num_processes == 1


def test_get_processes():
    process_list = ProcessList()
    process_list.add_processes([WhiteNoise(GENERATOR_LINSPACE)])
    assert process_list.get_processes(["white_noise"])[0].name == "white_noise"


def test_get_random_processes():
    process_list = ProcessList()
    process_list.add_processes([WhiteNoise(GENERATOR_LINSPACE)])
    assert len(process_list.get_random_processes(5)) == 5


if __name__ == "__main__":
    pytest.main()
