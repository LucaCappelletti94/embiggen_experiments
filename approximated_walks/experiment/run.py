from typing import Tuple

import pandas as pd
from embiggen import LinkPredictionTransformer
from ensmallen_graph import StringPPI, EnsmallenGraph
from tqdm.auto import tqdm, trange

from .multi_layer_perceptron import get_multi_layer_perceptron_predictions
from .skipgram import compute_skipgram_embedding


def run(
    root: str = "./",
    epochs: int = 100,
    batches_per_epoch: int = 2**10,
    embedding_size: int = 100,
    holdouts_number: int = 2,
    random_state: int = 42
) -> pd.DataFrame:
    results = []
    graph = StringPPI()
    negative_graph: EnsmallenGraph = graph.sample_negatives(
        graph.get_edges_number(),
        random_state=random_state,
        only_from_same_component=True
    )
    for holdout in trange(holdouts_number):
        pos_train, pos_test = graph.connected_holdout(
            train_size=0.8,
            random_state=random_state+holdout,
            verbose=False
        )
        pos_train.enable(
            vector_sources=True,
            vector_destinations=True,
            vector_outbounds=True
        )
        neg_train, neg_test = negative_graph.random_holdout(
            train_size=0.8,
            random_state=random_state+holdout,
            verbose=False
        )
        for max_neighbours in tqdm((10, pos_train.max_degree()), desc="Thresholds"):
            embedding = compute_skipgram_embedding(
                graph=pos_train,
                graph_name=graph.get_name(),
                holdout=holdout,
                root=root,
                embedding_size=embedding_size,
                epochs=epochs,
                max_neighbours=max_neighbours
            )
            transformer = LinkPredictionTransformer(
                method=None,
                aligned_node_mapping=True,
                support_mirror_strategy=True
            )
            x_train, y_train = transformer.transform(
                pos_train,
                neg_train,
            )
            x_test, y_test = transformer.transform(
                pos_test,
                neg_test,
            )
            train_metrics, test_metrics = get_multi_layer_perceptron_predictions(
                graph=pos_train,
                graph_name=graph.get_name(),
                embedding_method="SkipGram",
                max_neighbours=max_neighbours,
                holdout=holdout,
                root=root,
                embedding=embedding,
                batches_per_epoch=batches_per_epoch,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test
            )
            results.append({
                "run_type": "train",
                "embedding_model": "SkipGram",
                "Max Neighbours": max_neighbours,
                **train_metrics
            })
            results.append({
                "run_type": "test",
                "embedding_model": "SkipGram",
                "Max Neighbours": max_neighbours,
                **test_metrics
            })
    return pd.DataFrame(results)
