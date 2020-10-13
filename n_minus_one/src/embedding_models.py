from typing import Callable
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from embiggen import SkipGram, CBOW, GloVe, Node2VecSequence
import numpy as np
from cache_decorator import Cache


def build_node2vec_sequence(
    graph: EnsmallenGraph,
    walk_length: int,
    iterations: int,
    window_size: int,
    return_weight: float,
    explore_weight: float,
) -> Node2VecSequence:
    """Return sequence for training of Node2Vec-based models."""
    return Node2VecSequence(
        graph,
        walk_length=walk_length,
        batch_size=128,
        iterations=iterations,
        window_size=window_size,
        return_weight=return_weight,
        explore_weight=explore_weight
    )


@Cache(
    cache_path="results/embeddings/{graph_name}/skipgram/{_hash}.npy",
    args_to_ignore=["graph"]
)
def compute_skipgram_embeddings(
    graph: EnsmallenGraph,
    graph_name: str,
    graph_path: str,
    walk_length: int,
    iterations: int,
    window_size: int,
    return_weight: float,
    explore_weight: float,
    embedding_size: int,
    learning_rate: float,
    negative_samples: int,
    epochs: int
) -> np.ndarray:
    sequence = build_node2vec_sequence(
        graph,
        walk_length,
        iterations,
        window_size,
        return_weight,
        explore_weight
    )
    model = SkipGram(
        graph.get_nodes_number(),
        embedding_size=embedding_size,
        optimizer=Nadam(learning_rate),
        window_size=window_size,
        negative_samples=negative_samples
    )
    model.fit(
        sequence,
        epochs=epochs,
        verbose=0,
        callbacks=[
            EarlyStopping(monitor="loss", min_delta=0, patience=10)
        ]
    )
    return model.embedding


@Cache(
    cache_path="results/embeddings/{graph_name}/cbow/{_hash}.npy",
    args_to_ignore=["graph"]
)
def compute_cbow_embeddings(
    graph: EnsmallenGraph,
    graph_name: str,
    graph_path: str,
    walk_length: int,
    iterations: int,
    window_size: int,
    return_weight: float,
    explore_weight: float,
    embedding_size: int,
    learning_rate: float,
    negative_samples: int,
    epochs: int
) -> np.ndarray:
    sequence = build_node2vec_sequence(
        graph,
        walk_length,
        iterations,
        window_size,
        return_weight,
        explore_weight
    )
    model = CBOW(
        graph.get_nodes_number(),
        embedding_size=embedding_size,
        optimizer=Nadam(learning_rate),
        window_size=window_size,
        negative_samples=negative_samples
    )
    model.fit(
        sequence,
        epochs=epochs,
        verbose=0,
        callbacks=[
            EarlyStopping(monitor="loss", min_delta=0, patience=10)
        ]
    )
    return model.embedding


@Cache(
    cache_path="results/embeddings/{graph_name}/glove/{_hash}.npy",
    args_to_ignore=["graph"]
)
def compute_glove_embeddings(
    graph: EnsmallenGraph,
    graph_name: str,
    graph_path: str,
    walk_length: int,
    iterations: int,
    window_size: int,
    return_weight: float,
    explore_weight: float,
    embedding_size: int,
    learning_rate: float,
    epochs: int,
    **kwargs
) -> np.ndarray:
    words, contexts, frequencies = graph.cooccurence_matrix(
        walk_length,
        window_size=window_size,
        iterations=iterations,
        return_weight=return_weight,
        explore_weight=explore_weight,
        random_state=42,
        verbose=False
    )
    model = GloVe(
        graph.get_nodes_number(),
        embedding_size=embedding_size,
        optimizer=Nadam(learning_rate),
    )
    model.fit(
        (words, contexts),
        frequencies,
        epochs=epochs,
        batch_size=2**16,
        verbose=0,
        callbacks=[
            EarlyStopping(monitor="loss", min_delta=0, patience=10)
        ]
    )
    return model.embedding


AVAILABLE_MODELS = {
    "GloVe": compute_glove_embeddings,
    "CBOW": compute_cbow_embeddings,
    "SkipGram": compute_skipgram_embeddings
}


def get_embedding_model(embedding_model: str) -> Callable:
    """Return the embedding model curresponding to given name.

    Parameters
    ---------------------
    embedding_model: str,
        The name of the embedding model.

    Raises
    ---------------------
    ValueError,
        If given embedding model is not available.

    Returns
    ---------------------
    Callable to obtain embedding.
    """
    global AVAILABLE_MODELS
    if embedding_model not in AVAILABLE_MODELS:
        raise ValueError(
            (
                "Given embedding model is not available."
                "The available models are: {}."
            ).format(", ".join(AVAILABLE_MODELS.keys()))
        )
    return AVAILABLE_MODELS[embedding_model]
