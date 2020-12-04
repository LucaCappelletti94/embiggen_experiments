from typing import Generator
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
import os
import pandas as pd
from encodeproject import download
from tqdm.auto import trange


def load_complete_string_ppi(
    random_state: int = 42,
    holdouts_number: int = 10,
    train_size: float = 0.8
) -> Generator:
    """Return StringPPI graph."""
    path = "string_ppi.tsv.gz"
    tsv_path = path[:-3]
    url = "https://stringdb-static.org/download/protein.actions.v11.0/9606.protein.actions.v11.0.txt.gz"
    if not os.path.exists(path):
        download(url, path)
        pd.read_csv(path, sep="\t").to_csv(tsv_path, sep="\t")

    graph: EnsmallenGraph = EnsmallenGraph.from_unsorted_csv(
        tsv_path,
        directed=False,
        sources_column="item_id_a",
        destinations_column="item_id_b",
        weights_column="score",
        verbose=False
    )

    negative_graph: EnsmallenGraph = graph.sample_negatives(
        graph.get_edges_number(),
        random_state=random_state,
        only_from_same_component=True,
        verbose=False
    )
    for i in trange(holdouts_number, desc="Computing holdouts", leave=False):
        yield (
            i,
            *graph.connected_holdout(
                train_size=train_size,
                random_state=random_state+i,
                verbose=False
            ),
            *negative_graph.random_holdout(
                train_size=train_size,
                random_state=random_state+i,
                verbose=False
            )
        )
