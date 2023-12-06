from unittest import TestCase, main

from src.main.generator_linspace import GeneratorLinspace
from src.main.process.process_list import ProcessList
from src.main.process.white_noise_process import WhiteNoiseProcess

GENERATOR_LINSPACE = GeneratorLinspace(0.0, 1.0, 100)


class TestProcessList(TestCase):
    def test_add_processes(self):
        process_list = ProcessList()
        process_list.add_processes([WhiteNoiseProcess(GENERATOR_LINSPACE)])
        self.assertEqual(len(process_list.processes), 1)
        process_list.add_processes([WhiteNoiseProcess(GENERATOR_LINSPACE)])
        self.assertEqual(process_list.num_processes, 1)

    def test_get_processes(self):
        process_list = ProcessList()
        process_list.add_processes([WhiteNoiseProcess(GENERATOR_LINSPACE)])
        self.assertEqual(
            process_list.get_processes(["white_noise"])[0].name, "white_noise"
        )

    def test_get_random_processes(self):
        process_list = ProcessList()
        process_list.add_processes([WhiteNoiseProcess(GENERATOR_LINSPACE)])
        self.assertEqual(len(process_list.get_random_processes(5)), 5)


if __name__ == "__main__":
    main()
