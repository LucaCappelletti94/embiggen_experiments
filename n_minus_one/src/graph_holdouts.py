from typing import Tuple
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module


def connected_holdout(
    graph: EnsmallenGraph,
    train_size: float = 0.8,
    random_state: int = 42,
) -> Tuple[EnsmallenGraph]:
    """Return tuple containing the training and testing graphs.

    The resulting trainoing graph will have the same number of connected components
    as the original graph.

    Parameters
    ------------------------
    graph: EnsmallenGraph,
        The graph from which to extract holdout.
    train_size: float = 0.8,
        Rate of edges to leave for the validation.
    random_state: int = 42,
        The seed for making the holdout reproducible.

    Returns
    ------------------------
    Tuple with training and test graphs.
    """
    train, test = graph.connected_holdout(
        train_size=train_size,
        random_state=random_state,
        verbose=False
    )
    train.enable(
        vector_sources=True,
        vector_destinations=True,
        vector_outbounds=True
    )
    test.enable(
        vector_sources=True,
        vector_destinations=True,
        vector_outbounds=True
    )
    return train, test
