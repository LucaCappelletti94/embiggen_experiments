from tensorflow.keras.layers import Dense, BatchNormalization, ReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Accuracy, Recall, Precision, AUC
import pandas as pd
import numpy as np


def build_mlp_model() -> Model:
    """Return default model, a very simple MLP."""
    model = Sequential([
        Dense(128),
        BatchNormalization(),
        ReLU(),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=[
            Accuracy(),
            Recall(name="recall"),
            Precision(name="precision"),
            AUC(curve="ROC", name="auroc"),
            AUC(curve="PR", name="auprc"),
        ]
    )
    return model


def build_and_fit_mlp_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray
) -> pd.DataFrame:
    """Return results of training the default MLP model.

    Parameters
    -----------------------
    X_train: np.ndarray,
        The input data for the training set.
    X_test: np.ndarray,
        The input data for the test set.
    Y_train: np.ndarray,
        The labels for the train set.
    Y_test: np.ndarray,
        The labels for the test set.
    """
    model = build_mlp_model()
    history = model.fit(
        X_train,
        Y_train,
        batch_size=4096,
        epochs=100,
        verbose=0,
        callbacks=[
            EarlyStopping(monitor="auprc", min_delta=0.01, patience=5)
        ]
    )
    return pd.DataFrame(history)
