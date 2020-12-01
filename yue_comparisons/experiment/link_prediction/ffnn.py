from typing import Tuple
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input, Activation
from embiggen import LinkPredictionSequence
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
import pandas as pd
from cache_decorator import Cache


@Cache(
    cache_path="link_predictions/ffnn/{graph_name}/{holdout}_{_hash}.pkl.gz",
    args_to_ignore=["graph", "embedding"],
)
def compute_ffnn_predictions(
    graph: EnsmallenGraph,
    graph_name: str,
    holdout: int,
    embedding: pd.DataFrame,
    method: str = "Hadamard",
    negative_samples: float = 1.0,
    batch_size: int = 2**12,
    batches_per_epoch: int = 2**9,
    epochs: int = 1000,
    min_delta: int = 0.00001,
    patience: int = 5,
) -> Tuple[Sequential, pd.DataFrame]:
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
    method: str = "Hadamard",
        Method to use to compute the edge embedding.
    negative_samples: float = 1.0,
        Rate of negative samples to generate for each batch.
    batch_size: int = 2**12,
        Training batch size.
    batches_per_epoch: int = 2**9,
        Number of batches to generate per epoch.
    epochs: int = 1000,
        Maximum number of epochs to execute.
    min_delta: int = 0.00001,
        Minimum delta to wait for improvement of the loss function.
    patience: int = 5,
        Number of epochs to wait for an improvement.

    Returns
    ---------------------
    Training model and its training history.
    """
    # Creating the training sequence.
    sequence = LinkPredictionSequence(
        graph,
        embedding.values,
        method=method,
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        negative_samples=negative_samples,
    )
    # Creating the SkipGram model
    model = Sequential([
        Input(shape=(embedding.shape[1],)),
        Dense(128, activation="relu"),
        Dense(128, activation="linear"),
        BatchNormalization(),
        Activation("relu"),
        Dense(128, activation="relu"),
        Dense(128, activation="linear"),
        BatchNormalization(),
        Activation("relu"),
        Dense(64, activation="relu"),
        Dense(64, activation="linear"),
        BatchNormalization(),
        Activation("relu"),
        Dense(64, activation="relu"),
        Dense(64, activation="linear"),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(32, activation="linear"),
        BatchNormalization(),
        Activation("relu"),
        Dense(32, activation="relu"),
        Dense(32, activation="linear"),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(16, activation="linear"),
        BatchNormalization(),
        Activation("relu"),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    # Compiling the model
    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=[]  # TODO: ask tommy to publish the metrics
    )
    # Fitting the SkipGram model
    history = model.fit(
        sequence,
        steps_per_epoch=sequence.steps_per_epoch,
        callbacks=[
            EarlyStopping(
                "loss",
                min_delta=min_delta,
                patience=patience,
                mode="min"
            )
        ]
    ).history
    # Returning the obtained embedding
    return model, pd.DataFrame(history)
