from typing import Tuple
from embiggen.link_prediction import MultiLayerPerceptron
from ensmallen_graph import EnsmallenGraph
import pandas as pd
import numpy as np
from cache_decorator import Cache


@Cache(
    cache_path="{root}/link_predictions/multi_layer_perceptron/{embedding_method}/{graph_name}/{holdout}_{_hash}.pkl.gz",
    args_to_ignore=[
        "graph", "embedding",
        "x_train", "y_train",
        "x_test", "y_test"
    ],
)
def get_multi_layer_perceptron_predictions(
    graph: EnsmallenGraph,
    graph_name: str,
    embedding_method: str,
    holdout: int,
    embedding: pd.DataFrame,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    root: str,
    trainable: bool,
    method: str = "Concatenate",
    batches_per_epoch: int = 2**8
) -> Tuple[np.ndarray]:
    """Return trained model on given graph with training history.

    Parameters
    ----------------------
    graph: EnsmallenGraph,
        Graph to embed.
    graph_name: str,
        Name of the graph to embed.
    holdout: int,
        Number of the holdout to compute.
    embedding: pd.DataFrame,
        Pandas dataframe with the graph embedding.
    x_train: np.ndarray,
        Input data for computing the training performance.
    y_train: np.ndarray,
        Labels for computing the training performance.
    x_test: np.ndarray,
        Input data for computing the test performance.
    y_test: np.ndarray,
        Labels for computing the test performance.
    root: str,
        Where to store the results.
    method: str = "Concatenate",
        Method to use to compute the edge embedding.
    batches_per_epoch: int = 2**8,
        Number of batches to run for each epoch.

    Returns
    ---------------------
    Training model and its training history.
    """
    # Create new mlp
    mlp = MultiLayerPerceptron(
        embedding=embedding,
        edge_embedding_method=method,
        trainable=trainable
    )
    # Fit the mlp model
    mlp.fit(
        graph,
        batches_per_epoch=batches_per_epoch
    )
    # Computing predictions
    metric = (
        mlp.evaluate((*x_train.T, ), y_train, batch_size=2**16),
        mlp.evaluate((*x_test.T, ), y_test, batch_size=2**16),
    )
    return metric
