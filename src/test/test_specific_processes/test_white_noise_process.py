from unittest import TestCase, main

from src.main.specific_processes.white_noise_process import WhiteNoiseProcess


class TestWhiteNoiseProcess(TestCase):
    def test_generate_parameters(self):
        wnp = WhiteNoiseProcess((0.0, 1.0))
        parameters = wnp.generate_parameters()
        if parameters[0] == 1:
            self.assertGreaterEqual(parameters[2], parameters[1])

    def test_generate_time_series(self):
        wnp = WhiteNoiseProcess((0.0, 1.0))
        parameters = wnp.generate_parameters()
        ts, info = wnp.generate_time_series((5, parameters))
        self.assertEqual(len(ts.get_values()), 5)


if __name__ == "__main__":
    main()
