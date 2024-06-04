from dataclasses import dataclass

from networkx import Graph

from tsg.utils.typing import NDArrayFloat64T


@dataclass
class SourceData:
    def __init__(
        self,
        data_characteristics: NDArrayFloat64T | None = None,
        data_graph: Graph | None = None,
        shift: float = 0.0,
    ) -> None:
        self.data_characteristics = data_characteristics
        self.data_graph = data_graph
        self.shift = shift
