import numpy as np
import pytest
from hydra import compose, initialize

from tsg.linspace_info import LinspaceInfo
from tsg.scheduler.scheduler import Scheduler

GENERATOR_LINSPACE = LinspaceInfo(np.float64(0.0), np.float64(100.0), 100)


def test_generate_process_list():
    with initialize(version_base=None, config_path=".."):
        cfg = compose(config_name="config")
        schedule = Scheduler(cfg=cfg, linspace_info=GENERATOR_LINSPACE)
        assert (
            schedule.process_storage.processes["white_noise"] is not None
            and schedule.process_storage.processes["random_walk"] is not None
        )


def test_generate_steps_number():
    num_max = 100
    num_parts = 10
    strict_steps_list = Scheduler.generate_steps_number(num_max, num_parts, True)
    assert len(strict_steps_list) == num_parts
    assert sum(strict_steps_list) == num_max
    steps_list = Scheduler.generate_steps_number(num_max, num_parts)
    assert len(steps_list) <= num_parts
    assert sum(steps_list) == num_max


def test_generate_process_order():
    with initialize(version_base=None, config_path=".."):
        cfg = compose(config_name="config")
        schedule = Scheduler(cfg=cfg, linspace_info=GENERATOR_LINSPACE)
        process_order = schedule.generate_process_order()
        assert sum([steps for steps, _ in process_order]) == 100
        assert len(process_order) <= 10
        assert all(
            [
                process_name in schedule.process_storage.processes.keys()
                for _, process_name in process_order
            ]
        )


def test_generate_schedule():
    with initialize(version_base=None, config_path=".."):
        cfg = compose(config_name="config")
        scheduler = Scheduler(cfg=cfg, linspace_info=GENERATOR_LINSPACE)
        schedule = scheduler.generate_schedule()
        assert len(schedule) == len(scheduler.process_order)
        steps_sum = []
        for _, process_data in schedule:
            current_steps_sum = 0
            for steps, _ in process_data:
                current_steps_sum += steps
            steps_sum.append(current_steps_sum)
        assert all(
            steps_sum[i] == scheduler.process_order[i][0] for i in range(len(steps_sum))
        )


if __name__ == "__main__":
    pytest.main()
