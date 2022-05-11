"""Script providing experimental loop for the Node2Vec comparisons."""
import silence_tensorflow.auto
import os
from typing import Dict

import pandas as pd
from embiggen.pipelines import evaluate_embedding_for_edge_prediction
from ensmallen import Graph

from experiment import track_library
from tqdm.auto import tqdm
from experiment.data_retrieval import (retrieve_coo_ctd,
                                       retrieve_coo_pheknowlator,
                                       retrieve_coo_wikipedia)
from experiment.libraries import (
    GraPECPUSkipGramLibrary, GraPECPUCBOWLibrary, GraPEGPULibrary,
    FastNode2VecLibrary, PecanPyLibrary
)


def run_edge_prediction_evaluation_experiment():
    """Runs the edge prediction part of the experiments."""
    for LibraryBuilder, requires_sorting in tqdm((
        (GraPECPUSkipGramLibrary, False),
        (GraPECPUCBOWLibrary, False),
        (GraPEGPULibrary, True),
        (FastNode2VecLibrary, False),
        (PecanPyLibrary, False)
    ), desc="Libraries"):
        for graph_retrieval, edge_type, node_types, in tqdm(
            (
                (retrieve_coo_ctd, ["genes diseases"], ("genes", "diseases")),
                (retrieve_coo_pheknowlator, ["variant-disease"], ("VARIANT", "DISEASE")),
                (retrieve_coo_wikipedia, None, None)
            ),
            desc="Graphs"
        ):
            # If the graph is Wikipedia and the library is not GraPE CPU
            # we could not generate the embedding.
            if "GraPECPU" not in LibraryBuilder.get_library_name() and edge_type is None:
                continue

            def load_embedding(
                graph: Graph,
                holdout_number: int,
                **kwargs: Dict
            ) -> pd.DataFrame:
                """Execute computation of embedding of given graph."""
                if requires_sorting:
                    graph = graph.sort_by_decreasing_outbound_node_degree()
                # Reload the previously computed embedding
                embedding_path = track_library(
                    LibraryBuilder,
                    graph,
                    f"benchmarks/{holdout_number}"
                )
                # Load the embedding.
                return LibraryBuilder().load_embedding(graph, embedding_path)

            # Retrieve and create the current graph of interest
            graph = graph_retrieval()

            library_name = LibraryBuilder.get_library_name()

            # Create and check existance of the holdouts CSV performance report
            holdouts_path = os.path.join(
                "holdouts",
                library_name,
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

            for model_name in (
                "Perceptron",
                "DecisionTreeClassifier",
                "RandomForestClassifier",
                "LogisticRegression",
            ):
                # Compute the holdouts and histories given
                # the precomputed embedding
                holdouts = evaluate_embedding_for_edge_prediction(
                    load_embedding,
                    graph,
                    model=model_name,
                    edge_types=edge_type,
                    subgraph_of_interest_for_edge_prediction=subgraph_of_interest_for_edge_prediction,
                    sample_only_edges_with_heterogeneous_node_types=node_types is not None
                )

                # Storing the computed results
                os.makedirs(
                    os.path.join(
                        "holdouts",
                        library_name,
                    ),
                    exist_ok=True
                )
                holdouts.to_csv(holdouts_path, index=False)


if __name__ == "__main__":
    print("Starting edge prediction benchmarks experiments.")
    run_edge_prediction_evaluation_experiment()
