from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from embiggen import SkipGram, CBOW, GloVe, Node2VecSequence
import numpy as np


def build_node2vec_sequence(
    graph: EnsmallenGraph,
    walk_length: int,
    batch_size: int,
    iterations: int,
    window_size: int,
    return_weight: float,
    explore_weight: float,
) -> Node2VecSequence:
    """Return sequence for training of Node2Vec-based models."""
    return Node2VecSequence(
        graph,
        walk_length=walk_length,
        batch_size=batch_size,
        iterations=iterations,
        window_size=window_size,
        return_weight=return_weight,
        explore_weight=explore_weight
    )


def compute_skipgram_embeddings(
    graph: EnsmallenGraph,
    walk_length: int,
    batch_size: int,
    iterations: int,
    window_size: int,
    return_weight: float,
    explore_weight: float,
    embedding_size: int,
    learning_rate: float,
    negative_samples: int
) -> np.ndarray:
    sequence = build_node2vec_sequence(
        graph,
        walk_length,
        batch_size,
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
        epochs=100,
        callbacks=[
            EarlyStopping(monitor="loss", min_delta=0, patience=10)
        ]
    )
    return model.embedding


def compute_cbow_embeddings(
    graph: EnsmallenGraph,
    walk_length: int,
    batch_size: int,
    iterations: int,
    window_size: int,
    return_weight: float,
    explore_weight: float,
    embedding_size: int,
    learning_rate: float,
    negative_samples: int
) -> np.ndarray:
    sequence = build_node2vec_sequence(
        graph,
        walk_length,
        batch_size,
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
        epochs=100,
        callbacks=[
            EarlyStopping(monitor="loss", min_delta=0, patience=10)
        ]
    )
    return model.embedding


def compute_glove_embeddings(
    graph: EnsmallenGraph,
    walk_length: int,
    batch_size: int,
    iterations: int,
    window_size: int,
    return_weight: float,
    explore_weight: float,
    embedding_size: int,
    learning_rate: float
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
        epochs=100,
        callbacks=[
            EarlyStopping(monitor="loss", min_delta=0, patience=10)
        ]
    )
    return model.embedding
