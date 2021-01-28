from typing import Tuple, Dict, List

import pandas as pd
from embiggen import LinkPredictionTransformer
from ensmallen_graph import EnsmallenGraph
import os
from ensmallen_graph.datasets.string import HumanString
from tqdm.auto import tqdm, trange
from multiprocessing import cpu_count, Pool

from .multi_layer_perceptron import get_multi_layer_perceptron_predictions
from .skipgram import compute_skipgram_embedding
import tensorflow as tf


def get_gpus() -> List["LogicalDevice"]:
    """Return list of detected GPUs."""
    return tf.config.experimental.list_physical_devices('GPU')


def get_gpu_number() -> int:
    """Return number of available GPUs."""
    return len(get_gpus())


def enable_subgpu_training():
    """Enable subgpu training using tensorflow."""
    for gpu in get_gpus():
        tf.config.experimental.set_memory_growth(gpu, True)


def subrun(
    holdout: int,
    root: str = "./",
    epochs: int = 100,
    batches_per_epoch: int = 2**10,
    embedding_size: int = 100,
    random_state: int = 42,
) -> Dict:
    enable_subgpu_training()
    os.environ["ROCR_VISIBLE_DEVICES"] = str(holdout % get_gpu_number())
    graph = HumanString()
    negative_graph: EnsmallenGraph = graph.sample_negatives(
        graph.get_edges_number(),
        random_state=random_state,
        only_from_same_component=True
    )
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
    results = []
    for max_neighbours in tqdm((10, pos_train.max_degree()), desc="Thresholds", leave=False):
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
    return results


def subrun_wrapper(kwargs):
    return subrun(**kwargs)


def run(
    root: str = "./",
    epochs: int = 100,
    batches_per_epoch: int = 2**10,
    embedding_size: int = 100,
    holdouts_number: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    processes = get_gpu_number()
    enable_subgpu_training()
    with Pool(processes) as p:
        results = pd.DataFrame([
            perf
            for perfs in tqdm(
                p.imap(
                    subrun_wrapper,
                    (
                        dict(
                            holdout=holdout,
                            root="./",
                            epochs=100,
                            batches_per_epoch=2**10,
                            embedding_size=100,
                            random_state=42
                        )
                        for holdout in range(holdouts_number)
                    )
                ),
                total=holdouts_number,
                desc="Computing holdouts"
            )
            for perf in perfs
        ])
        p.close()
        p.join()

    return results
