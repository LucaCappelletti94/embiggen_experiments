"""Script providing experimental loop for the Node2Vec comparisons."""
import os
import sys
from typing import Dict

import pandas as pd
import silence_tensorflow.auto
from embiggen.pipelines import evaluate_embedding_for_edge_prediction
from ensmallen import Graph

from experiment import track_library
from experiment.data_retrieval import (retrieve_coo_ctd,
                                       retrieve_coo_pheknowlator)
from experiment.libraries import FastNode2VecLibrary


def run_fastnode2vec_embedding(
    graph: Graph,
    holdout_number: int,
    **kwargs: Dict
) -> pd.DataFrame:
    """Execute computation of embedding of given graph."""
    embedding_path = track_library(
        FastNode2VecLibrary, graph, f"benchmarks/{holdout_number}")
    return FastNode2VecLibrary().load_embedding(graph, embedding_path)


def run_fastnode2vec_embedding_experiment():
    """Runs the embedding part of the experiments.

    To use the minimum required amount of (expensive) computation from Google Cloud,
    we split the execution of the benchmarks (the computation of the Node2Vec embeddings)
    and the evaluation of their link prediction performance through a Perceptron by
    executing them on different machines.
    """
    for graph_retrieval, edge_type in (
        (retrieve_coo_ctd, "genes diseases"),
        (retrieve_coo_pheknowlator, "variant-disease"),
    ):
        graph = graph_retrieval()
        evaluate_embedding_for_edge_prediction(
            run_fastnode2vec_embedding,
            graph,
            model_name="Perceptron",
            edge_types=[edge_type],
            only_execute_embeddings=True
        )


if __name__ == "__main__":
    print("Starting fastnode2vec node embedding benchmarks experiments.")
    run_fastnode2vec_embedding_experiment()
