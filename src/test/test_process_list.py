from unittest import TestCase, main

from src.main.process_list import ProcessList
from src.main.specific_processes.white_noise_process import WhiteNoiseProcess

BORDER_VALUES = (-10, 10)


class TestProcessList(TestCase):
    def test_add_processes(self):
        process_list = ProcessList()
        process_list.add_processes([WhiteNoiseProcess(BORDER_VALUES)])
        self.assertEqual(len(process_list.processes), 1)
        process_list.add_processes([WhiteNoiseProcess(BORDER_VALUES)])
        self.assertEqual(process_list.num_processes, 1)

    def test_get_processes(self):
        process_list = ProcessList()
        process_list.add_processes([WhiteNoiseProcess(BORDER_VALUES)])
        self.assertEqual(
            process_list.get_processes(["white_noise"])[0].name, "white_noise"
        )

    def test_get_random_processes(self):
        process_list = ProcessList()
        process_list.add_processes([WhiteNoiseProcess(BORDER_VALUES)])
        self.assertEqual(len(process_list.get_random_processes(5)), 5)


if __name__ == "__main__":
    main()
