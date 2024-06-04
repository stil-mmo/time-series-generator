import numpy as np
from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.source_data_sampling.point_clustering import get_border_value, move_points
from tsg.source_data_sampling.source_data import SourceData
from tsg.source_data_sampling.source_data_sampling_method import (
    SourceDataSamplingMethod,
)
from tsg.utils.typing import NDArrayFloat64T


class SurfaceSamplingMethod(SourceDataSamplingMethod):
    name = "surface_sampling_method"

    def __init__(self, linspace_info_cfg: DictConfig) -> None:
        super().__init__(linspace_info_cfg=linspace_info_cfg)

    def sample_source_data(self, num_samples: int) -> tuple[SourceData, LinspaceInfo]:
        coordinates = self.sample_spherical(num_samples)
        shift = move_points(coordinates)
        border_values = (
            get_border_value(coordinates, is_min=True),
            get_border_value(coordinates, is_min=False),
        )
        linspace_info = self.get_linspace_info(
            start=border_values[0], stop=border_values[1]
        )
        source_data = SourceData(
            data_characteristics=np.transpose(coordinates), shift=shift
        )
        return source_data, linspace_info

    @staticmethod
    def sample_spherical(num_points: int, ndim=3) -> NDArrayFloat64T:
        vec = np.random.randn(ndim, num_points)
        vec /= np.linalg.norm(vec, axis=0)
        return vec
