"""Module providing APIs towards SNAP Node2Vec."""
from ensmallen import Graph
import pandas as pd
import subprocess

__library_name__ = "SNAP"


def build_graph(
    graph: Graph,
    path: str
):
    """Creates graph object to be used for SNAP.

    Parameters
    -------------------------
    graph: Graph
        The graph to be made available for SNAP.
    path: str
        The path where to store the graph.
    """
    pd.DataFrame(
        graph.get_directed_edge_node_ids()
    ).to_csv(
        path,
        sep=" ",
        index=False,
        header=False
    )


def compute_node_embedding(
    path: str,
    embedding_path: str,
    random_walk_length: int,
    iterations_per_node: int,
    epochs: int,
    p: float,
    q: float,
):
    """Compute node embedding using SNAP's Node2Vec.
    
    Parameters
    -------------------------
    path: str
        Path from where to load the graph
    embedding_path: str
        Path where to store the embedding
    random_walk_length: int
        Length of the random walk
    iterations_per_node: int
        Number of iterations to execute per node
    epochs: int
        Number of epochs to run the embedding for
    p: float
        Value of the explore weight
    q: float
        Value of the return weight
    """
    subprocess.run([
        "node2vec",
        "-i:{}".format(path),
        "-o:{}".format(embedding_path),
        "-l:{}".format(random_walk_length),
        "-r:{}".format(iterations_per_node),
        "-e:{}".format(epochs),
        "-p:{}".format(p),
        "-q:{}".format(q),
        # Treat the graph as undirected.
        "-dr:NO",
        # Treat the graph as unweighted.
        "-w:NO",
        # Do not show the loading bar as it may slow
        # down the execution of the library during the benchmarks.
        "-v:NO"
    ])


def load_embedding(
    graph: Graph,
    embedding_path: str,
) -> pd.DataFrame:
    """Returns embedding computed from the SNAP Node2Vec implementation.
    
    Parameters
    --------------------------
    graph: Graph
        The graph associated to the embedding.
    embedding_path: str
        The path from where to load the embedding.
    """
    # Load the Embedding from the provided path.
    embedding = pd.read_csv(
        embedding_path,
        # This specific CSV uses space as a separator.
        sep=" ",
        # Skip the first line which contains metadata
        skiprows=1,
        # Use the first colum as index.
        index_col=0,
        # The CSV does not have a header.
        header=None
    )
    # Reindex the embedding.
    embedding.index = [
        graph.get_node_name_from_node_id(node_id)
        for node_id in embedding.index.values
    ]
    # Reindex it to make sure it is aligned with provided graph.
    snap_embedding = embedding.loc[graph.get_node_names()]
    # Return the loaded embedding.
    return snap_embedding