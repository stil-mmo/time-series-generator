import numpy as np
from numpy.typing import NDArray
from src.main.generator_linspace import GeneratorLinspace
from src.main.scheduler.scheduler import Scheduler
from src.main.source_data_processing.aggregated_data import AggregatedData


class SchedulerStorage:
    def __init__(
        self,
        num_steps: int,
        generator_linspace: GeneratorLinspace,
        points: NDArray[np.float32],
        clusters: NDArray[int],
    ):
        self.num_steps = num_steps
        self.generator_linspace = generator_linspace
        self.points = points
        self.clusters = clusters
        self.scheduler_storage = self.create_storage()

    def create_storage(self) -> dict[int, Scheduler]:
        storage = {}
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
