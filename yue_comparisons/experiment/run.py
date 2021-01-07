from typing import Callable

import pandas as pd
from embiggen import LinkPredictionTransformer
from tqdm.auto import tqdm
import compress_json

from .link_prediction import (get_multi_layer_perceptron_predictions,
                              load_link_prediction_graphs)


def run(
    embedding_model: Callable,
    root: str = "./",
    epochs: int = 1000,
    batches_per_epoch: int = 2**14,
    embedding_size: int = 100,
    holdouts_number: int = 10,
    verbose: bool = False
) -> pd.DataFrame:
    methods = {
        "compute_cbow_embedding_cached": "Grape CBOW",
        "compute_glove_embedding_cached": "Grape GloVe",
        "compute_skipgram_embedding_cached": "Grape SkipGram",
    }
    results = []
    parameters = compress_json.local_load("parameters.json")
    for holdout_number, graph_name, pos_train, pos_test, neg_train, neg_test in load_link_prediction_graphs(
        holdouts_number=holdouts_number,
        verbose=verbose
    ):
        embedding = embedding_model(
            graph=pos_train,
            graph_name=graph_name,
            holdout=holdout_number,
            root=root,
            **parameters[graph_name],
            embedding_size=embedding_size,
            epochs=epochs
        )
        transformer = LinkPredictionTransformer(
            method=None,
            aligned_node_mapping=True
        )
        x_train, y_train = transformer.transform(
            pos_train,
            neg_train,
        )
        x_test, y_test = transformer.transform(
            pos_test,
            neg_test,
        )
        for trainable in tqdm((True, False), desc="Trainable", leave=False):
            train_perf, test_perf = get_multi_layer_perceptron_predictions(
                graph=pos_train,
                graph_name=pos_train.get_name(),
                embedding_method=embedding_model.__name__,
                holdout=holdout_number,
                root=root,
                embedding=embedding,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                trainable=trainable,
                batches_per_epoch=batches_per_epoch
            )
            results.append({
                "run_type": "train",
                "graph": graph_name,
                "trainable": trainable,
                "method": methods[embedding_model.__name__],
                **train_perf
            })
            results.append({
                "run_type": "test",
                "graph": graph_name,
                "trainable": trainable,
                "method": methods[embedding_model.__name__],
                **test_perf
            })
    return pd.DataFrame(results)
