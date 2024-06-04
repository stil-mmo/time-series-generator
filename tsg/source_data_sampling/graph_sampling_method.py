import networkx as nx
import numpy as np
from networkx import Graph
from omegaconf import DictConfig

from tsg.linspace_info import LinspaceInfo
from tsg.source_data_sampling.source_data import SourceData
from tsg.source_data_sampling.source_data_sampling_method import (
    SourceDataSamplingMethod,
)
from tsg.utils.typing import NDArrayFloat64T


class GraphSamplingMethod(SourceDataSamplingMethod):
    name = "graph_sampling_method"

    def __init__(self, linspace_info_cfg: DictConfig) -> None:
        super().__init__(linspace_info_cfg=linspace_info_cfg)

    def sample_source_data(
        self, num_samples: int, dim: int = 3
    ) -> tuple[SourceData, LinspaceInfo]:
        num_edges = np.random.randint(num_samples, num_samples * 2)
        graph = self.sample_graph(num_samples, num_edges)
        coordinates = self.get_nodes_coordinates(num_samples, graph, dim)
        source_data = SourceData(data_graph=graph, data_characteristics=coordinates)
        linspace_info = self.get_linspace_info()
        return source_data, linspace_info

    @staticmethod
    def sample_graph(num_vertices: int, num_edges: int) -> Graph:
        return nx.bull_graph()

    @staticmethod
    def get_nodes_coordinates(
        num_samples: int, graph: Graph, dim: int = 3
    ) -> NDArrayFloat64T:
        coordinates: NDArrayFloat64T = np.ndarray(shape=(num_samples, dim))
        pos = nx.spring_layout(graph, dim=dim)
        for node in pos.keys():
            coordinates[node] = pos[node]
        return coordinates
