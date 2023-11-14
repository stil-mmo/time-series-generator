from abc import ABC, abstractmethod

from numpy import array

from src.main.time_series import TimeSeries


class Process(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        return "base_process"

    @property
    @abstractmethod
    def lag(self) -> int:
        return 0

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        return 0

    @abstractmethod
    def generate_parameters(
        self, low_value: float, high_value: float
    ) -> tuple[float, ...]:
        return tuple()

    @abstractmethod
    def generate_init_values(self, low_value: float, high_value: float) -> array:
        return array([])

    @abstractmethod
    def get_info(
        self, sample: tuple[int, tuple], init_values: tuple[float, ...] = None
    ) -> dict:
        return dict()

    @abstractmethod
    def generate_time_series(
        self,
        sample: tuple[int, tuple],
        previous_values: array = None,
        border_values: tuple[float, float] = None,
    ) -> tuple[TimeSeries, dict]:
        return TimeSeries(), self.get_info(sample)

    @abstractmethod
    def draw_plot(
        self,
        border_values: tuple[float, float] = None,
        path: str = None,
        time_series_data: tuple[TimeSeries, dict] = None,
    ):
        return
