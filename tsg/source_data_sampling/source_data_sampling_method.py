from abc import ABC, abstractmethod

from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.source_data_sampling.source_data import SourceData


class SourceDataSamplingMethod(ABC):
    name = ""

    def __init__(self, linspace_info_cfg: DictConfig) -> None:
        self.linspace_info_cfg = linspace_info_cfg

    @abstractmethod
    def sample_source_data(self, num_samples: int) -> tuple[SourceData, LinspaceInfo]:
        pass

    def get_linspace_info(
        self, start: float | None = None, stop: float | None = None
    ) -> LinspaceInfo:
        return LinspaceInfo(
            start=self.linspace_info_cfg.linspace_borders[0]
            if start is None
            else start,
            stop=self.linspace_info_cfg.linspace_borders[1] if stop is None else stop,
            parts=self.linspace_info_cfg.linspace_parts,
            center_shift=self.linspace_info_cfg.center_shift,
            step_coeff=self.linspace_info_cfg.step_coeff,
            use_k=self.linspace_info_cfg.use_k,
        )
