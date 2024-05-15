from dataclasses import dataclass

from tsg.utils.typing import GraphT, NDArrayFloat64T


@dataclass
class SourceData:
    def __init__(
        self,
        data_characteristics: NDArrayFloat64T | None = None,
        data_graph: GraphT | None = None,
        shift: float = 0.0,
    ) -> None:
        self.data_characteristics = data_characteristics
        self.data_graph = data_graph
        self.shift = shift
