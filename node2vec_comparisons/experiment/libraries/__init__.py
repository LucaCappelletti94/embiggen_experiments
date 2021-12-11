"""Submodule providing wrappers for different libraries APIs."""
from .grape_library import GraPELibrary
from .graph_embedding_library import GraphEmbeddingLibrary
from .node2vec_library import Node2VecLibrary
from .nodevectors_library import NodeVectorsLibrary
from .pecanpy_library import PecanPyLibrary
from .snap_library import SNAPLibrary

libraries = [
    GraPELibrary,
    GraphEmbeddingLibrary,
    Node2VecLibrary,
    NodeVectorsLibrary,
    PecanPyLibrary,
    SNAPLibrary
]

__all__ = [
    "GraPELibrary",
    "GraphEmbeddingLibrary",
    "Node2VecLibrary",
    "NodeVectorsLibrary",
    "PecanPyLibrary",
    "SNAPLibrary"
]
