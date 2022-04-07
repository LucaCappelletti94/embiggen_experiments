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
from experiment.libraries import PecanPyLibrary


def run_pecanpy_embedding(
    graph: Graph,
    holdout_number: int,
    **kwargs: Dict
) -> pd.DataFrame:
    """Execute computation of embedding of given graph."""
    embedding_path = track_library(
        PecanPyLibrary, graph, f"benchmarks/{holdout_number}")
    return PecanPyLibrary().load_embedding(graph, embedding_path)


def run_pecanpy_embedding_experiment():
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
            run_pecanpy_embedding,
            graph,
            model_name="Perceptron",
            edge_types=[edge_type],
            only_execute_embeddings=True
        )


def run_pecanpy_edge_prediction_experiment():
    """Runs the edge prediction part of the experiments."""
    for graph_retrieval, edge_type in (
        (retrieve_coo_ctd, "genes diseases"),
        (retrieve_coo_pheknowlator, "variant-disease"),
    ):
        # Retrieve and create the current graph of interest
        graph = graph_retrieval()

        # Create and check existance of the holdouts CSV performance report
        holdouts_path = os.path.join(
            "holdouts",
            "PecanPy",
            f"{graph.get_name()}.csv"
        )

        # If the holdouts were already computed
        if os.path.exists(holdouts_path):
            continue

        # Create the subgraph of interest for the task,
        # which in the context of CTD and PheKnowLator
        # is the portion of the graph with the edge type
        # of interest.
        subgraph_of_interest_for_edge_prediction = graph.filter_from_names(
            edge_type_names_to_keep=[edge_type]
        )

        # Compute the holdouts and histories given
        # the precomputed embedding
        holdouts, histories = evaluate_embedding_for_edge_prediction(
            run_pecanpy_embedding,
            graph,
            model_name="Perceptron",
            edge_types=[edge_type],
            only_execute_embeddings=False,
            subgraph_of_interest_for_edge_prediction=subgraph_of_interest_for_edge_prediction
        )

        # Storing the computed results
        os.makedirs(
            os.path.join(
                "holdouts",
                "PecanPy",
            ),
            exist_ok=True
        )
        holdouts.to_csv(holdouts_path, index=False)

        # Storing the training histories
        for i, history in enumerate(histories):
            os.makedirs(
                os.path.join(
                    "histories",
                    "PecanPy",
                    graph.get_name()
                ),
                exist_ok=True
            )
            history.to_csv(os.path.join(
                "histories",
                "PecanPy",
                graph.get_name(),
                f"{i}.csv",
            ), index=False)


if __name__ == "__main__":
    if sys.argv[1] == "edge_prediction":
        print("Starting PecanPy edge prediction experiments.")
        run_pecanpy_edge_prediction_experiment()
    else:
        print("Starting PecanPy node embedding benchmarks experiments.")
        run_pecanpy_embedding_experiment()
