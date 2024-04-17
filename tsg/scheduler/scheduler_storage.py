import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.parameters_generation.parameters_generation_method import (
    ParametersGenerationMethod,
)
from tsg.scheduler.scheduler import Scheduler


class SchedulerStorage:
    def __init__(
        self,
        cfg: DictConfig,
        linspace_info: LinspaceInfo,
        points: NDArray[np.float64],
        clusters: NDArray[np.float64],
    ):
        self.cfg = cfg
        self.linspace_info = linspace_info
        self.points = points
        self.clusters = clusters
        self.scheduler_storage = self.create_storage()

    def create_storage(self) -> dict[int, Scheduler]:
        storage: dict[int, Scheduler] = {}
        for cluster in self.clusters:
            if cluster not in storage.keys():
                scheduler = Scheduler(
                    cfg=self.cfg,
                    linspace_info=self.linspace_info,
                )
                storage[cluster] = scheduler
        return storage

    def get_cluster(self, point_index: int) -> int:
        return self.clusters[point_index]

    def get_scheduler(
        self,
        cluster: int,
        parameters_generation_method: ParametersGenerationMethod,
    ) -> Scheduler:
        self.scheduler_storage[
            cluster
        ].parameters_generation_method = parameters_generation_method
        return self.scheduler_storage[cluster]
