import numpy as np
from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.source_data_sampling.source_data import SourceData
from tsg.source_data_sampling.source_data_sampling_method import (
    SourceDataSamplingMethod,
)
from tsg.utils.typing import GraphT


class GraphSamplingMethod(SourceDataSamplingMethod):
    name = "graph_sampling_method"

    def __init__(self, linspace_info_cfg: DictConfig) -> None:
        super().__init__(linspace_info_cfg=linspace_info_cfg)

    def sample_source_data(self, num_samples: int) -> tuple[SourceData, LinspaceInfo]:
        num_edges = np.random.randint(num_samples, num_samples * 2)
        graph = self.sample_graph(num_samples, num_edges)
        source_data = SourceData(data_graph=graph)
        linspace_info = self.get_linspace_info()
        return source_data, linspace_info

    @staticmethod
    def sample_graph(num_vertices: int, num_edges: int) -> GraphT:
        vertices = np.arange(num_vertices)
        graph: GraphT = dict()
        for i in range(num_vertices):
            graph[i] = set()

        for i in range(num_edges):
            u = np.random.randint(0, num_vertices)
            v = np.random.choice(vertices[vertices != u])
            graph[u].add(v)
            graph[v].add(u)

        return graph
