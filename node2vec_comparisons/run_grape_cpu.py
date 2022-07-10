"""Script providing experimental loop for the Node2Vec comparisons."""
import os
import sys
from typing import Dict

import pandas as pd
from embiggen.pipelines import evaluate_embedding_for_edge_prediction
from ensmallen import Graph

from experiment import track_library
from tqdm.auto import tqdm
from experiment.data_retrieval import (retrieve_coo_ctd,
                                       retrieve_coo_pheknowlator,
                                       retrieve_coo_wikipedia)
from experiment.libraries import GraPECPUSkipGramLibrary, GraPECPUCBOWLibrary


def run_grape_skipgram_embedding(
    graph: Graph,
    holdout_number: int,
    **kwargs: Dict
) -> pd.DataFrame:
    """Execute computation of embedding of given graph."""
    embedding_path = track_library(
        GraPECPUSkipGramLibrary, graph, f"benchmarks/{holdout_number}")
    return GraPECPUSkipGramLibrary().load_embedding(graph, embedding_path)


def run_grape_cbow_embedding(
    graph: Graph,
    holdout_number: int,
    **kwargs: Dict
) -> pd.DataFrame:
    """Execute computation of embedding of given graph."""
    embedding_path = track_library(
        GraPECPUCBOWLibrary, graph, f"benchmarks/{holdout_number}")
    return GraPECPUCBOWLibrary().load_embedding(graph, embedding_path)


def run_grape_embedding_experiment():
    """Runs the embedding part of the experiments.

    To use the minimum required amount of (expensive) computation from Google Cloud,
    we split the execution of the benchmarks (the computation of the Node2Vec embeddings)
    and the evaluation of their link prediction performance through a Perceptron by
    executing them on different machines.

    """
    for library_callback in tqdm((
        run_grape_skipgram_embedding,
        run_grape_cbow_embedding
    ), desc="Libraries"):
        for graph_retrieval, edge_type in tqdm((
            (retrieve_coo_ctd, ["genes diseases"]),
            (retrieve_coo_pheknowlator, ["variant-disease"]),
            (retrieve_coo_wikipedia, None)
        ), desc="Graphs"):
            graph = graph_retrieval()
            evaluate_embedding_for_edge_prediction(
                library_callback,
                graph,
                model_name="Perceptron",
                edge_types=edge_type,
                only_execute_embeddings=True
            )


if __name__ == "__main__":
    print("Starting GraPE CPU node embedding benchmarks experiments.")
    run_grape_embedding_experiment()
