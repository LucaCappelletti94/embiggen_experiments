"""Module providing tools to benchmark libraries."""
import gc
import os
from time import sleep
from typing import Callable, Union

import compress_json
from ensmallen import Graph
from memory_time_tracker import Tracker
from tqdm.auto import tqdm

from .libraries import libraries


def wait_k_seconds(k: int):
    """Sleeps for given amount of seconds running the garbage collector each time.

    Parameters
    -----------------------
    k: int,
        Number of seconds to sleep for.
    """
    for _ in range(k):
        sleep(1)
        # Should not be necessary but apparently it is.
        gc.collect()


def track_library(
    library: Callable,
    graph: Union[Graph, str],
    root: str
):
    """Execute tracking of provided library on given graph.

    Parameters
    -----------------------
    library: Callable
        The library to benchmark.
    graph: Union[Graph, str]
        The graph to benchmark with.
    root: str
        Root of the directory where to store the results.
    """
    tracker_log_path = os.path.join(
        root,
        "tracker",
        library.get_library_name(),
        "{}.csv".format(graph if isinstance(graph, str) else graph.get_name())
    )
    edge_list_path = os.path.join(
        root,
        "edge_list",
        library.get_library_name(),
        "{}.edge".format(graph if isinstance(graph, str) else graph.get_name())
    )
    embedding_path = os.path.join(
        root,
        "embedding",
        library.get_library_name(),
        "{}.csv".format(graph if isinstance(graph, str) else graph.get_name())
    )
    if os.path.exists(embedding_path):
        return embedding_path
    for path in (tracker_log_path, edge_list_path, embedding_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    lib = library()
    if isinstance(graph, Graph):
        lib.store_graph(graph, edge_list_path)
    with Tracker(tracker_log_path):
        lib.compute_node_embedding(
            edge_list_path,
            embedding_path,
            **compress_json.local_load("parameters.json")
        )

    wait_k_seconds(20)
    return embedding_path


def track_all_libraries(
    graph: Union[Graph, str],
    root: str,
    disable_alias_method: bool = False
):
    """Run benchmark of all libraries.

    Parameters
    -----------------------
    graph: Union[Graph, str]
        The graph to benchmark with.
    root: str
        Root of the directory where to store the results.
    disable_alias_method: bool = False
        Whether to skip execution of Alias-method based libraries.
    """
    for library in tqdm(
        libraries,
        desc="Libraries",
        leave=False,
        dynamic_ncols=True
    ):
        if disable_alias_method:
            if library.get_library_name() in ("GraphEmbedding", "SNAP", "Node2Vec"):
                continue
        track_library(
            library,
            graph,
            root,
        )
