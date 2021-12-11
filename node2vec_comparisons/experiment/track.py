"""Module providing tools to benchmark libraries."""
import gc
import os
from time import sleep
from typing import Callable

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
    graph: Graph,
    root: str
):
    """Execute tracking of provided library on given graph.

    Parameters
    -----------------------
    library: Callable
        The library to benchmark.
    graph: Graph
        The graph to benchmark with.
    root: str
        Root of the directory where to store the results.
    """
    tracker_log_path = os.path.join(
        root,
        "tracker",
        library.get_library_name(),
        graph.get_name(),
    )
    edge_list_path = os.path.join(
        root,
        "edge_list",
        library.get_library_name(),
        graph.get_name()
    )
    embedding_path = os.path.join(
        root,
        "edge_list",
        library.get_library_name(),
        graph.get_name()
    )
    if os.path.exists(tracker_log_path):
        return
    lib = library()
    lib.store_graph(graph, edge_list_path)
    with Tracker(tracker_log_path):
        lib.compute_node_embedding(
            edge_list_path,
            embedding_path,
            **compress_json.local_load("parameters.json")
        )

    wait_k_seconds(seconds)


def track_all_libraries(
    graph: Graph,
    root: str,
):
    """Run benchmark of all libraries.

    Parameters
    -----------------------
    graph: Graph
        The graph to benchmark with.
    root: str
        Root of the directory where to store the results.
    """
    for library in tqdm(
        libraries,
        desc="Libraries",
        leave=False,
        dynamic_ncols=True
    ):
        track_library(
            library,
            graph,
            root,
        )
