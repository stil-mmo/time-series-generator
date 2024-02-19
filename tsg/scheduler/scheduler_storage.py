import numpy as np
from numpy.typing import NDArray

from tsg.linspace_info import LinspaceInfo
from tsg.sampling.aggregated_data import AggregatedData
from tsg.scheduler.scheduler import Scheduler


class SchedulerStorage:
    def __init__(
        self,
        num_steps: int,
        generator_linspace: LinspaceInfo,
        points: NDArray,
        clusters: NDArray,
    ):
        self.num_steps = num_steps
        self.generator_linspace = generator_linspace
        self.points = points
        self.clusters = clusters
        self.scheduler_storage = self.create_storage()

    def create_storage(self) -> dict[int, Scheduler]:
        storage: dict[int, Scheduler] = {}
        for cluster in self.clusters:
            if cluster not in storage.keys():
                scheduler = Scheduler(self.num_steps, self.generator_linspace)
                storage[cluster] = scheduler
        return storage

    def get_cluster(self, point_index: int) -> int:
        return self.clusters[point_index]

    def get_scheduler(
        self, cluster: int, aggregated_data: AggregatedData | None = None
    ) -> Scheduler:
        if aggregated_data is not None:
            self.scheduler_storage[cluster].set_aggregated_data(aggregated_data)
        return self.scheduler_storage[cluster]