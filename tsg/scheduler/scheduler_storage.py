from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.process.process_storage import ProcessStorage
from tsg.scheduler.scheduler import Scheduler
from tsg.utils.typing import NDArrayFloat64T


class SchedulerStorage:
    def __init__(
        self,
        num_steps: int,
        cfg_scheduler: DictConfig,
        linspace_info: LinspaceInfo,
        process_storage: ProcessStorage,
        points: NDArrayFloat64T,
        clusters: NDArrayFloat64T,
    ) -> None:
        self.num_steps = num_steps
        self.cfg_scheduler = cfg_scheduler
        self.linspace_info = linspace_info
        self.process_storage = process_storage
        self.points = points
        self.clusters = clusters
        self.scheduler_storage = self.create_storage()

    def create_storage(self) -> dict[int, Scheduler]:
        storage: dict[int, Scheduler] = {}
        for cluster in self.clusters:
            if cluster not in storage.keys():
                scheduler = Scheduler(
                    num_steps=self.num_steps,
                    linspace_info=self.linspace_info,
                    process_storage=self.process_storage,
                    strict_num_parts=self.cfg_scheduler.strict_num_parts,
                    stable_parameters=self.cfg_scheduler.stable_parameters,
                    process_order=self.cfg_scheduler.process_order,
                )
                storage[cluster] = scheduler
        return storage

    def get_cluster(self, point_index: int) -> int:
        return self.clusters[point_index]

    def get_scheduler(self, cluster: int) -> Scheduler:
        return self.scheduler_storage[cluster]
