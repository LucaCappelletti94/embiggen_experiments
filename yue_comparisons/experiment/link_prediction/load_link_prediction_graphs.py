import os
from glob import glob
from typing import Generator, Tuple

import pandas as pd
from ensmallen_graph.datasets.yue import CTDDDA, NDFRTDDA, DrugBankDDI, StringPPI
from tqdm.auto import tqdm, trange


def load_link_prediction_graphs(
    root: str = "data",
    train_size: float = 0.8,
    holdouts_number: int = 10,
    random_state: int = 42,
    verbose: bool = False
) -> Generator:
    """Lazily yields graphs to execute on link prediction.

    Parameters
    ---------------------
    root: str = "data",
        Directory from where to load the graphs.
    train_size: float = 0.8,
        Rate of the training graph edges to reserve for validation.
    holdouts_number: int = 10,
        Number of the holdouts to execute.
    random_state: int = 42,
        Random state of the holdouts.
    verbose: bool = False,
        Wether to show loading bar

    Returns
    ---------------------
    Quadruple with number of the holdout and the positive and negative train and test graphs.
    """
    for graph_builder in tqdm((StringPPI, CTDDDA, DrugBankDDI, NDFRTDDA), desc="Graphs", leave=False):
        graph = graph_builder(verbose=verbose)
        negative_graph: EnsmallenGraph = graph.sample_negatives(
            graph.get_edges_number(),
            random_state=random_state,
            only_from_same_component=True,
            verbose=verbose
        )
        for i in trange(holdouts_number, desc="Computing holdouts for graph {}".format(graph.get_name()), leave=False):
            train, test = graph.connected_holdout(
                train_size=train_size,
                random_state=random_state+i,
                verbose=verbose
            )
            train.enable(
                vector_sources=True,
                vector_destinations=True,
                vector_outbounds=True
            )
            yield (
                i,
                graph.get_name(),
                train,
                test,
                *negative_graph.random_holdout(
                    train_size=train_size,
                    random_state=random_state+i,
                    verbose=verbose
                )
            )
