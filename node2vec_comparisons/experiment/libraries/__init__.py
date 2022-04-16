"""Submodule providing wrappers for different libraries APIs."""
from .grape_gpu_library import GraPEGPULibrary
from .grape_cpu_library import GraPECPULibrary
from .graph_embedding_library import GraphEmbeddingLibrary
from .node2vec_library import Node2VecLibrary
from .nodevectors_library import NodeVectorsLibrary
from .pecanpy_library import PecanPyLibrary
from .snap_library import SNAPLibrary
from .fastnode2vec_library import FastNode2VecLibrary

libraries = [
    GraPEGPULibrary,
    GraPECPULibrary,
    GraphEmbeddingLibrary,
    Node2VecLibrary,
    NodeVectorsLibrary,
    PecanPyLibrary,
    FastNode2VecLibrary,
    SNAPLibrary
]

__all__ = [
    "GraPEGPULibrary",
    "GraPECPULibrary",
    "GraphEmbeddingLibrary",
    "Node2VecLibrary",
    "NodeVectorsLibrary",
    "PecanPyLibrary",
    "FastNode2VecLibrary",
    "SNAPLibrary"
]
