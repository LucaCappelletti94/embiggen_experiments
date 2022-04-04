"""Script providing experimental loop for the Node2Vec comparisons."""
from typing import Dict
import silence_tensorflow.auto
from experiment.data_retrieval import retrieve_coo_ctd, retrieve_coo_pheknowlator, retrieve_coo_wikipedia
from experiment import track_library
from experiment.libraries import PecanPyLibrary, GraPELibrary
from embiggen.pipelines import evaluate_embedding_for_edge_prediction
from ensmallen import Graph
import pandas as pd


def run_pecanpy_embedding(
    graph: Graph,
    holdout_number: int,
    **kwargs: Dict
) -> pd.DataFrame:
    """Execute computation of embedding of given graph."""
    embedding_path = track_library(
        PecanPyLibrary, graph, f"benchmark/{holdout_number}")
    return PecanPyLibrary().load_embedding(graph, embedding_path)


def run_pecanpy_embedding_experiment():
    """Runs the embedding part of the experiments.

    To use the minimum required amount of (expensive) computation from Google Cloud,
    we split the execution of the benchmarks (the computation of the Node2Vec embeddings)
    and the evaluation of their link prediction performance through a Perceptron by
    executing them on different machines.

    """
    for graph_retrieval, edge_type in (
        (retrieve_coo_ctd, "chem gene ixns"),
        (retrieve_coo_pheknowlator, "variant-disease"),
    ):
        graph = graph_retrieval()
        evaluate_embedding_for_edge_prediction(
            run_pecanpy_embedding,
            graph,
            model_name="Perceptron",
            edge_types=[edge_type],
            only_execute_embeddings=True
        )


if __name__ == "__main__":
    run_pecanpy_embedding_experiment()
