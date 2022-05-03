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


def run_grape_edge_prediction_experiment():
    """Runs the edge prediction part of the experiments."""
    for graph_retrieval, edge_type, node_types in (
        (retrieve_coo_ctd, ["genes diseases"], ("VARIANT", "DISEASE")),
        (retrieve_coo_pheknowlator, [
         "variant-disease"], ("genes", "diseases")),
        (retrieve_coo_wikipedia, None, None)
    ):
        # Retrieve and create the current graph of interest
        graph = graph_retrieval()

        # Create and check existance of the holdouts CSV performance report
        holdouts_path = os.path.join(
            "holdouts",
            "GraPE",
            f"{graph.get_name()}.csv"
        )

        # If the holdouts were already computed
        if os.path.exists(holdouts_path):
            continue

        # Create the subgraph of interest for the task,
        # which in the context of CTD and PheKnowLator
        # is the portion of the graph with the edge type
        # of interest.
        if node_types is not None:
            subgraph_of_interest_for_edge_prediction = graph.filter_from_names(
                node_type_name_to_keep=node_types
            )
        else:
            # For Wikipedia we do not care to focus the edge prediction
            # task on a specific portion of the graph, instead, we want to
            # run the edge prediction task on the entire graph.
            subgraph_of_interest_for_edge_prediction = None

        # Compute the holdouts and histories given
        # the precomputed embedding
        holdouts, histories = evaluate_embedding_for_edge_prediction(
            run_grape_embedding,
            graph,
            model_name="Perceptron",
            edge_types=edge_type,
            only_execute_embeddings=False,
            subgraph_of_interest_for_edge_prediction=subgraph_of_interest_for_edge_prediction,
            sample_only_edges_with_heterogeneous_node_types=True
        )

        # Storing the computed results
        os.makedirs(
            os.path.join(
                "holdouts",
                "GraPE",
            ),
            exist_ok=True
        )
        holdouts.to_csv(holdouts_path, index=False)

        # Storing the training histories
        for i, history in enumerate(histories):
            os.makedirs(
                os.path.join(
                    "histories",
                    "GraPE",
                    graph.get_name()
                ),
                exist_ok=True
            )
            history.to_csv(os.path.join(
                "histories",
                "GraPE",
                graph.get_name(),
                f"{i}.csv",
            ), index=False)


if __name__ == "__main__":
    if sys.argv[1] == "edge_prediction":
        print("Starting GraPE CPU edge prediction experiments.")
        run_grape_edge_prediction_experiment()
    else:
        print("Starting GraPE CPU node embedding benchmarks experiments.")
        run_grape_embedding_experiment()
