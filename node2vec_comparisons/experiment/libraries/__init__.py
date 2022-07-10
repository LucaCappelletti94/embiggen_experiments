"""Submodule providing wrappers for different libraries APIs."""
from .grape_gpu_library import GraPEGPULibrary
from .grape_cpu_cbow_library import GraPECPUCBOWLibrary
from .grape_cpu_skipgram_library import GraPECPUSkipGramLibrary
from .graph_embedding_library import GraphEmbeddingLibrary
from .node2vec_library import Node2VecLibrary
from .nodevectors_library import NodeVectorsLibrary
from .pecanpy_library import PecanPyLibrary
from .snap_library import SNAPLibrary
from .fastnode2vec_library import FastNode2VecLibrary

libraries = [
    GraPEGPULibrary,
    GraPECPUCBOWLibrary,
    GraPECPUSkipGramLibrary,
    GraphEmbeddingLibrary,
    Node2VecLibrary,
    NodeVectorsLibrary,
    PecanPyLibrary,
    FastNode2VecLibrary,
    SNAPLibrary
]

__all__ = [
    "GraPEGPULibrary",
    "GraPECPUCBOWLibrary",
    "GraPECPUSkipGramLibrary",
    "GraphEmbeddingLibrary",
    "Node2VecLibrary",
    "NodeVectorsLibrary",
    "PecanPyLibrary",
    "FastNode2VecLibrary",
    "SNAPLibrary"
]
