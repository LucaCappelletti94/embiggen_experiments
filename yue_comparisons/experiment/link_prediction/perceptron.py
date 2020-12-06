from typing import Tuple
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from embiggen import LinkPredictionSequence
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
import pandas as pd
import numpy as np
from cache_decorator import Cache
from tensorflow.keras.metrics import AUC


@Cache(
    cache_path="{root}/link_predictions/perceptron/{graph_name}/{holdout}_{_hash}.pkl.gz",
    args_to_ignore=["graph", "embedding", "x_train", "x_test", "y_test"],
)
def get_perceptron_predictions(
    graph: EnsmallenGraph,
    graph_name: str,
    holdout: int,
    embedding: pd.DataFrame,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    root: str,
    method: str = "Concatenate",
    negative_samples: float = 2.0,
    batch_size: int = 2**12,
    batches_per_epoch: int = 2**15,
    epochs: int = 1000,
    min_delta: int = 0.00001,
    patience: int = 5,
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
    x_test: np.ndarray,
        Input data for computing the test performance.
    root: str,
        Where to store the results.
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
        Input(shape=(embedding.shape[1]*2,)),
        Dense(1, activation="sigmoid"),
    ])
    # Compiling the model
    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            AUC(curve="PR", name="AUPRC"),
            AUC(curve="ROC", name="AUROC")
        ]
    )
    # Fitting the SkipGram model
    model.fit(
        sequence,
        steps_per_epoch=sequence.steps_per_epoch,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[
            EarlyStopping(
                "loss",
                min_delta=min_delta,
                patience=patience,
                mode="min"
            )
        ]
    )
    # Compute the model predictions
    train_pred = model.predict(x_train, batch_size=2**12)
    test_pred = model.predict(x_test, batch_size=2**12)
    return train_pred, test_pred
