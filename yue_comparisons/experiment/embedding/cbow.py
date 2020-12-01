from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Nadam
from embiggen import CBOW, Node2VecSequence
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
import pandas as pd
from cache_decorator import Cache


@Cache(
    cache_path="{root}/embeddings/cbow/{graph_name}/{holdout}_{_hash}.csv.xz",
    args_to_ignore=["graph"],
)
def compute_cbow_embedding(
    graph: EnsmallenGraph,
    graph_name: str,
    holdout: int,
    root: str,
    walk_length: int = 100,
    batch_size: int = 256,
    iterations: int = 20,
    window_size: int = 4,
    return_weight: float = 1.0,
    explore_weight: float = 1.0,
    embedding_size: int = 100,
    negative_samples: int = 10,
    epochs: int = 1000,
    min_delta: int = 0.1,
    patience: int = 5,
    learning_rate: float = 0.1
) -> pd.DataFrame:
    """Return embedding computed using CBOW on given graph.

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
    walk_length: int = 100,
        Length of the random walks.
    batch_size: int = 256,
        Dimension of the batch size.
    iterations: int = 20,
        Number of iterations per node.
    window_size: int = 4,
        Dimension of the window size for the context.
    return_weight: float = 1.0,
        Value for the return weight, inverse of the p parameter.
    explore_weight: float = 1.0,
        Value for the explore weight, inverse of the q parameter.
    embedding_size: int = 100,
        Dimension of the embedding.
    negative_samples: int = 10,
        Number of negative samples to extract using the NCE loss.
    epochs: int = 1000,
        Maximum number of epochs to execute.
    min_delta: int = 0.00001,
        Minimum delta to wait for improvement of the loss function.
    patience: int = 5,
        Number of epochs to wait for an improvement.
    learning_rate: float = 0.1,
        Learning rate to use with the Nadam optimizer.

    Returns
    ---------------------
    Pandas dataframe with the computed embedding.
    """
    # Creating the training sequence.
    sequence = Node2VecSequence(
        graph,
        walk_length=walk_length,
        batch_size=batch_size,
        iterations=iterations,
        window_size=window_size,
        return_weight=return_weight,
        explore_weight=explore_weight,
        support_mirror_strategy=False
    )
    # Creating the CBOW model
    model = CBOW(
        graph.get_nodes_number(),
        embedding_size=embedding_size,
        window_size=window_size,
        negative_samples=negative_samples
    )
    # Fitting the CBOW model
    model.fit(
        sequence,
        steps_per_epoch=sequence.steps_per_epoch,
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
