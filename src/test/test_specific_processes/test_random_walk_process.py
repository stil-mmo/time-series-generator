from unittest import TestCase, main

from src.main.generator_linspace import GeneratorLinspace
from src.main.specific_processes.random_walk_process import RandomWalkProcess

GENERATOR_LINSPACE = GeneratorLinspace(0.0, 1.0, 100)


def check_random_walk(array):
    res = True
    for i in range(1, len(array)):
        if array[i] != array[i - 1] + 1 and array[i] != array[i - 1] - 1:
            res = False
            break
    return res


class TestRandomWalkProcess(TestCase):
    def test_generate_parameters(self):
        up, down, walk = RandomWalkProcess(GENERATOR_LINSPACE).generate_parameters()
        self.assertTrue(up + down == 1.0)

    def test_generate_init_values(self):
        init_value = RandomWalkProcess(GENERATOR_LINSPACE).generate_init_values()
        self.assertTrue(-1 <= init_value[0] <= 1)

    def test_generate_time_series(self):
        rw_time_series, info = RandomWalkProcess(
            GENERATOR_LINSPACE
        ).generate_time_series((10, (0.5, 0.5, 1.0)))
        self.assertTrue(len(rw_time_series.get_values()) == 10)
        self.assertTrue(check_random_walk(rw_time_series.get_values()))


if __name__ == "__main__":
    main()
