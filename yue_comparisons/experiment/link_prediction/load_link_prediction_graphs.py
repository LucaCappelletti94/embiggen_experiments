from typing import Generator, Tuple
import os
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from ..utils import get_link_prediction_paths
from tqdm.auto import tqdm, trange
from glob import glob
import pandas as pd


def load_link_prediction_graphs(
    root: str = "data",
    train_size: float = 0.8,
    holdouts_number: int = 10,
    random_state: int = 42,
    verbose: bool = False
) -> Generator:
    """Lazily yields graphs to execute on link prediction.

    Parameters
    ---------------------
    root: str = "data",
        Directory from where to load the graphs.
    train_size: float = 0.8,
        Rate of the training graph edges to reserve for validation.
    holdouts_number: int = 10,
        Number of the holdouts to execute.
    random_state: int = 42,
        Random state of the holdouts.
    verbose: bool = False,
        Wether to show loading bar

    Returns
    ---------------------
    Quadruple with number of the holdout and the positive and negative train and test graphs.
    """
    for path in glob("{}/*/*.tsv*.xz".format(root)):
        if not os.path.exists(path[:-3]):
            pd.read_csv(path, sep="\t").to_csv(
                path[:-3],
                sep="\t",
                index=False
            )
    for path in tqdm(get_link_prediction_paths(root), desc="Graphs", leave=False):
        edge_path = "{}/edge_list.tsv".format(path)
        node_path = "{}/node_list.tsv".format(path)
        graph_name = path.split(os.sep)[-1]
        graph: EnsmallenGraph = EnsmallenGraph.from_unsorted_csv(
            edge_path=edge_path,
            directed=False,
            node_path=node_path,
            numeric_edge_node_ids=True,
            sources_column="subject",
            destinations_column="object",
            weights_column="weight",
            nodes_column="id",
            skip_weights_if_unavailable=True,
            
            name=graph_name,
            verbose=verbose
        )
        negative_graph: EnsmallenGraph = graph.sample_negatives(
            graph.get_edges_number(),
            random_state=random_state,
            only_from_same_component=True,
            verbose=verbose
        )
        for i in trange(holdouts_number, desc="Computing holdouts for graph {}".format(graph.get_name()), leave=False):
            train, test = graph.connected_holdout(
                train_size=train_size,
                random_state=random_state+i,
                verbose=verbose
            )
            train.enable(
                vector_sources=True,
                vector_destinations=True,
                vector_outbounds=True
            )
            yield (
                i,
                graph_name,
                train,
                test,
                *negative_graph.random_holdout(
                    train_size=train_size,
                    random_state=random_state+i,
                    verbose=verbose
                )
            )
