"""Load the graph and its features."""
from typing import Callable, Tuple
import pandas as pd
from ensmallen_graph import EnsmallenGraph
from ensmallen_graph.datasets.linqs.parse_linqs import get_words_data


def load_graph_and_features(graph_loader: Callable) -> Tuple[EnsmallenGraph, pd.DataFrame]:
    """Return graph and its features.

    Parameters
    ------------------------
    graph_loader: Callable,
        Method to use to load the graphs.

    Returns
    ------------------------
    Tuple with loaded graph and its features.
    """
    complete_graph: EnsmallenGraph = graph_loader(verbose=False)

    graph_without_words: EnsmallenGraph = complete_graph.remove(
        deny_node_types_set=set(["Word", "Unknown"]),
        verbose=False
    ).remove(
        singletons=True,
        selfloops=True,
        verbose=False
    )

    return graph_without_words, get_words_data(complete_graph).loc[graph_without_words.get_node_names()]
