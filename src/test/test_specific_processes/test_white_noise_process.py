from unittest import TestCase, main

from src.main.generator_linspace import GeneratorLinspace
from src.main.process.white_noise_process import WhiteNoiseProcess

GENERATOR_LINSPACE = GeneratorLinspace(0.0, 1.0, 100)


class TestWhiteNoiseProcess(TestCase):
    def test_generate_parameters(self):
        wnp = WhiteNoiseProcess(GENERATOR_LINSPACE)
        parameters = wnp.generate_parameters()
        if parameters[0] == 1:
            self.assertGreaterEqual(parameters[2], parameters[1])

    def test_generate_time_series(self):
        wnp = WhiteNoiseProcess(GENERATOR_LINSPACE)
        parameters = wnp.generate_parameters()
        ts, info = wnp.generate_time_series((5, parameters))
        self.assertEqual(len(ts.get_values()), 5)


if __name__ == "__main__":
    main()
