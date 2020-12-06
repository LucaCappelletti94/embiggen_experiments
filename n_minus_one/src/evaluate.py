import numpy as np
from typing import Tuple
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from embiggen import GraphTransformer
from cache_decorator import Cache
from .load_graph import load_graph
from .embedding_models import get_embedding_model
from .mlp_model import build_and_fit_mlp_model
from .graph_holdouts import connected_holdout


def get_training_data(
    graph: EnsmallenGraph,
    train: EnsmallenGraph,
    test: EnsmallenGraph,
    edge_embedding_method: str,
    embedding: np.ndarray,
    train_size: float,
    random_state: int
) -> Tuple[np.ndarray]:
    # Genrate positive examples
    transformer = GraphTransformer(edge_embedding_method)
    transformer.fit(embedding)
    pos_X_train = transformer.transform(train)
    pos_X_test = transformer.transform(test)
    # Generate negative examples
    negatives: EnsmallenGraph = graph.sample_negatives(
        graph.get_edges_number(),
        random_state=random_state,
        verbose=False
    )
    neg_train, neg_test = negatives.random_holdout(
        train_size,
        random_state=random_state,
        verbose=False
    )
    neg_X_train = transformer.transform(neg_train)
    neg_X_test = transformer.transform(neg_test)
    # Create train values and labels
    y_train = np.hstack((
        np.ones(pos_X_train.shape[0]), np.zeros(neg_X_train.shape[0])
    ))
    X_train = np.hstack((
        pos_X_train,
        neg_X_train
    ))
    # Shuffle the indices
    indices = np.arange(0, X_train.shape[0])
    rng = np.random.RandomState(seed=random_state)  # pylint: disable=no-member
    rng.shuffle(indices)
    # Apply the same shffule to both vectors
    y_train = y_train[indices]
    X_train = X_train[indices]

    # Create test values and labels
    y_test = np.hstack((
        np.ones(pos_X_test.shape[0]), np.zeros(neg_X_test.shape[0])
    ))
    X_test = np.hstack((
        pos_X_test,
        neg_X_test
    ))
    # Shuffle the indices
    indices = np.arange(0, X_test.shape[0])
    rng.shuffle(indices)
    # Apply the same shffule to both vectors
    y_test = y_test[indices]
    X_test = X_test[indices]

    return (X_train, X_test, y_train, y_test)


@Cache(
    cache_path="{results_folder}/{graph_name}/{embedding_model}/{_hash}.json"
)
def evaluate(
    results_folder: str,
    graph_name: str,
    graph_path: str,
    has_weights: bool,
    embedding_model: str,
    edge_embedding_method: str,
    walk_length: int,
    iterations: int,
    window_size: int,
    return_weight: float,
    explore_weight: float,
    embedding_size: int,
    learning_rate: float,
    negative_samples: int,
    mlp_epochs: int,
    embedder_epochs: int,
    train_size: float = 0.8,
    random_state: int = 42,
):
    """Return the performance of the MLP model on the link prediction task with the given parameters.

    Parameters
    --------------------------
    graph_path: str,
        Path from where to load the graph.
    has_weights: bool,
        Wether the graph has weights.
    embedding_model: str,
        The embedding model to use.
    edge_embedding_method: str,
        The edge embedding method to use.
    walk_length: int,
        The length of the random walks.
    iterations: int,
        Number of walks to execute from each node.
    window_size: int,
        The distance to use to consider nodes as contextual.
    return_weight: float,
        The weight to use to increase the locality of the walk.
        This is the same as the inverse of the p parameter.
    explore_weight: float,
        The weight to use to make the walk more explorative.
        This is the same as the inverse of the q parameter.
    embedding_size: int,
        The dimension of the embeddings.
    learning_rate: float,
        The learning rate for the model.
    negative_samples: int,
        The number of negative samples, if it applies.
    mlp_epochs: int,
        The maximum number of training epochs for the mlp model.
    embedder_epochs: int,
        The maximum number of training epochs for the ebedder model.
    train_size: float = 0.8,
        The rate of edges to allocate for the training.
    random_state: int = 42,
        The random state to use for the experiment.

    Returns
    -------------------------
    Dictionary with the model performance and parameters.
    """
    # Loading the graph from given file
    graph = load_graph(graph_path, has_weights)
    # Computing connected holdout
    train, test = connected_holdout(
        graph, random_state=random_state
    )
    # Retrieving the embedder model
    embedder = get_embedding_model(embedding_model)
    # Computing graph embedding
    embedding = embedder(
        train,
        graph_name=graph_name,
        graph_path=graph_path,
        walk_length=walk_length,
        iterations=iterations,
        window_size=window_size,
        return_weight=return_weight,
        explore_weight=explore_weight,
        embedding_size=embedding_size,
        learning_rate=learning_rate,
        negative_samples=negative_samples,
        epochs=embedder_epochs,
    )
    # Retrieving training data
    X_train, X_test, y_train, y_test = get_training_data(
        graph,
        train,
        test,
        edge_embedding_method,
        embedding,
        train_size,
        random_state
    )

    # Computing performance of MLP model on Link Prediction task
    performance = build_and_fit_mlp_model(
        X_train, X_test, y_train, y_test, epochs=mlp_epochs)

    return dict(
        graph_name=graph_name,
        graph_path=graph_path,
        has_weights=has_weights,
        embedding_model=embedding_model,
        edge_embedding_method=edge_embedding_method,
        walk_length=walk_length,
        iterations=iterations,
        window_size=window_size,
        return_weight=return_weight,
        explore_weight=explore_weight,
        embedding_size=embedding_size,
        learning_rate=learning_rate,
        negative_samples=negative_samples,
        mlp_epochs=mlp_epochs,
        embedder_epochs=embedder_epochs,
        train_size=train_size,
        random_state=random_state,
        **performance.iloc[-1].to_dict()
    )
