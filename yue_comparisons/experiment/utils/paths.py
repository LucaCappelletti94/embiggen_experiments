from typing import List
from glob import glob
import os


def get_link_prediction_paths(root: str = "data") -> List[str]:
    """Return path to directories of the graph to be used for link prediction.

    Parameters
    -----------------------
    root: str = "data",
        Directory from where to search for link prediction data.

    Returns
    -----------------------
    List of paths to the directories.
    """
    return [
        graph_directory
        for graph_directory in glob("{}/*".format(root))
        if len(glob("{}/*.tsv".format(graph_directory))) == 2
    ]


def get_node_label_paths(root: str = "data") -> List[str]:
    """Return path to directories of the graph to be used for node label prediction.

    Parameters
    -----------------------
    root: str = "data",
        Directory from where to search for node label prediction data.

    Returns
    -----------------------
    List of paths to the directories.
    """
    return [
        graph_directory
        for graph_directory in glob("{}/*".format(root))
        if len(glob("{}/*.tsv".format(graph_directory))) > 2
    ]