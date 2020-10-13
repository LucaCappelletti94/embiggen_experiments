from tensorflow.keras.layers import Dense, BatchNormalization, ReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.metrics import Accuracy, Recall, Precision, AUC

def build_mlp_model()->Model:
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