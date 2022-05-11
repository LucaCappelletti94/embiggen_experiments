"""Script providing experimental loop for the Node2Vec comparisons."""
import os
import sys
from typing import Dict

import pandas as pd
import silence_tensorflow.auto
from embiggen.pipelines import evaluate_embedding_for_edge_prediction
from ensmallen import Graph
from embiggen.utils import has_gpus

from experiment import track_library
from experiment.data_retrieval import (retrieve_coo_ctd,
                                       retrieve_coo_pheknowlator,
                                       retrieve_coo_wikipedia)
from experiment.libraries import GraPEGPULibrary


def run_grape_embedding(
    graph: Graph,
    holdout_number: int,
    **kwargs: Dict
) -> pd.DataFrame:
    """Execute computation of embedding of given graph."""
    graph = graph.sort_by_decreasing_outbound_node_degree()
    embedding_path = track_library(
        GraPEGPULibrary, graph, f"benchmarks/{holdout_number}")
    return GraPEGPULibrary().load_embedding(graph, embedding_path)


def run_grape_embedding_experiment():
    """Runs the embedding part of the experiments.

    To use the minimum required amount of (expensive) computation from Google Cloud,
    we split the execution of the benchmarks (the computation of the Node2Vec embeddings)
    and the evaluation of their link prediction performance through a Perceptron by
    executing them on different machines.

    """
    for graph_retrieval, edge_type in (
        (retrieve_coo_ctd, ["genes diseases"]),
        (retrieve_coo_pheknowlator, ["variant-disease"]),
        (retrieve_coo_wikipedia, None)
    ):
        graph = graph_retrieval()
        evaluate_embedding_for_edge_prediction(
            run_grape_embedding,
            graph,
            model_name="Perceptron",
            edge_types=edge_type,
            only_execute_embeddings=True
        )


if __name__ == "__main__":
    if not has_gpus():
        raise ValueError(
            "It does not make sense to run the benchmarks of this library "
            "on a machine without GPUs."
        )
    print("Starting GraPE node embedding benchmarks experiments.")
    run_grape_embedding_experiment()
