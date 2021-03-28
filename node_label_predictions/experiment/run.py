"""Main loop of the node label prediction experiment.

Node-label prediction prediction
=================================================
In this experiment with run the Node-Label prediction Neighbours-based model,
appreviated to NoLaN.

Used graphs
------------------------------
We use Cora, CiteSeer and Pubmed as made available from the

"""
from typing import Dict, Tuple
import pandas as pd
from ensmallen_graph import EnsmallenGraph
from ensmallen_graph.utils import holdouts_generator
from ensmallen_graph.datasets.linqs import Cora, CiteSeer, PubMedDiabetes
from embiggen.utils import get_available_node_embedding_methods, compute_node_embedding
from embiggen.node_prediction import NoLaN
import compress_json
from cache_decorator import Cache
from .utils import load_graph_and_features


@Cache(
    cache_path=[
        "nolan/{node_embedding_method_name}/{graph_name}/{holdout_number}_{_hash}_training_history.csv.xz",
        "nolan/{node_embedding_method_name}/{graph_name}/{holdout_number}_{_hash}_performance.csv.xz",
    ]
)
def evaluate_nolan_performance(
    train_graph: EnsmallenGraph,
    validation_graph: EnsmallenGraph,
    node_embedding: pd.DataFrame,
    node_features: pd.DataFrame,
    node_embedding_method_name: str,
    graph_name: str,
    holdout_number: int,
    configuration: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return.

    Parameters
    ---------------------------------
    train_graph: EnsmallenGraph,
        The training graph.
    validation_graph: EnsmallenGraph,
        The validation graph.
    node_embedding: pd.DataFrame,
        Embedding of the graph.
    node_features: pd.DataFrame,
        Node features of the graph's nodes.
    node_embedding_method_name: str,
        Name of the node embedding method.
    graph_name: str,
        Name of the graph.
    holdout_number: int,
        Number of the holdout.
    configuration: Dict,
        Configuration for the NoLaN model.

    Returns
    ---------------------------------
    Tuple with training history and performance.
    """
    nolan = NoLaN(
        train_graph,
        node_embedding,
        node_features,
        **configuration
    )
    node_label_history = nolan.fit(
        train_graph,
        validation_graph=validation_graph
    )
    return node_label_history, nolan.evaluate(train_graph, validation_data=validation_graph)


def run_node_label_prediction():
    """Run the node-label prediction."""
    node_embedding_configuration = compress_json.local_load(
        "node_embedding_configuration.json"
    )
    nolan_configuration = compress_json.local_load(
        "nolan_configuration.json"
    )
    all_performance = []
    for graph, node_features in (
        load_graph_and_features(graph_loader)
        for graph_loader in (Cora, CiteSeer, PubMedDiabetes)
    ):
        graph_name = graph.get_name()
        for node_embedding_method_name in get_available_node_embedding_methods():
            _, node_embedding = compute_node_embedding(
                graph,
                node_embedding_method_name=node_embedding_method_name,
                explore_weight=10.0,
                return_weight=0.01,
                **node_embedding_configuration[graph_name]
            )
            for holdout_number, (train, validation) in holdouts_generator(graph.node_label_holdout):
                _, performance = evaluate_nolan_performance(
                    train,
                    validation,
                    node_embedding,
                    node_features,
                    node_embedding_method_name,
                    graph_name,
                    holdout_number,
                    nolan_configuration[graph_name]
                )
                performance["graph_name"] = graph_name
                performance["holdout_number"] = holdout_number
                performance["node_embedding_method_name"] = node_embedding_method_name

                all_performance.append(performance)
    return pd.concat(all_performance)
