from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module


def load_graph(path: str, has_weights: bool) -> EnsmallenGraph:
    """Load graph from given path using Ensmallen.

    Parameters
    ----------------------
    path: str,
        Path from where to load the graph.
    has_weights: bool,
        Wethever to load the weights.

    Returns
    ----------------------
    Loaded graph.
    """
    return EnsmallenGraph.from_unsorted_csv(
        path,
        directed=False,
        sources_column="subject",
        destinations_column="object",
        **(dict(
            weights_column="weight"
        ) if has_weights else {}),
        verbose=False
    )
