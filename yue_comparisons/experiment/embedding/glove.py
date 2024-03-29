from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Nadam
from embiggen import GloVe, Node2VecSequence
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
import pandas as pd
from cache_decorator import Cache


@Cache(
    cache_path="{root}/embeddings/glove/{graph_name}/{holdout}_{_hash}.csv.xz",
    args_to_ignore=["graph"],
)
def compute_glove_embedding(
    graph: EnsmallenGraph,
    graph_name: str,
    holdout: int,
    root: str,
    return_weight: float,
    explore_weight: float,
    walk_length: int = 128,
    batch_size: int = 2**18,
    iterations: int = 20,
    window_size: int = 4,
    embedding_size: int = 100,
    alpha: int = 0.75,
    epochs: int = 100,
    min_delta: int = 0.0001,
    patience: int = 5,
    learning_rate: float = 0.01
) -> pd.DataFrame:
    """Return embedding computed using GloVe on given graph.

    Parameters
    ----------------------
    graph: EnsmallenGraph,
        Graph to embed.
    graph_name: str,
        Name of the graph to embed.
    holdout: int,
        Number of the holdout to compute.
    root: str,
        Where to store the results.
    return_weight: float,
        Value for the return weight, inverse of the p parameter.
    explore_weight: float,
        Value for the explore weight, inverse of the q parameter.
    walk_length: int = 128,
        Length of the random walks.
    batch_size: int = 2**18,
        Dimension of the batch size.
    iterations: int = 20,
        Number of iterations per node.
    window_size: int = 4,
        Dimension of the window size for the context.
    embedding_size: int = 100,
        Dimension of the embedding.
    alpha: float = 0.75,
        Coefficient for GloVe.
    epochs: int = 100,
        Maximum number of epochs to execute.
    min_delta: int = 0.0001,
        Minimum delta to wait for improvement of the loss function.
    patience: int = 5,
        Number of epochs to wait for an improvement.
    learning_rate: float = 0.1,
        Learning rate to use with the Nadam optimizer.

    Returns
    ---------------------
    Pandas dataframe with the computed embedding.
    """
    # Computing the co-occurrence matrix
    sources, destinations, frequencies = graph.cooccurence_matrix(
        walk_length,
        window_size=window_size,
        iterations=iterations,
        return_weight=return_weight,
        explore_weight=explore_weight
    )
    # Creating the GloVe model
    model = GloVe(
        graph.get_nodes_number(),
        embedding_size=embedding_size,
        alpha=alpha,
        optimizer=Nadam(learning_rate=learning_rate)
    )
    # Fitting the GloVe model
    model.fit(
        (sources, destinations),
        frequencies,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            EarlyStopping(
                "loss",
                min_delta=min_delta,
                patience=patience,
                mode="min"
            )
        ]
    )
    # Returning the obtained embedding
    return model.get_embedding_dataframe(graph.get_node_names())
