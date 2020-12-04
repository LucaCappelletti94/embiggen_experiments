from typing import Tuple
from embiggen import LinkPredictionTransformer
import pandas as pd
from .perceptron import get_perceptron_predictions
from .skipgram import compute_skipgram_embedding
from .load_complete_string_ppi import load_complete_string_ppi
from .metrics import compute_metrics
from tqdm.auto import tqdm


def run(
    root: str = "./",
    epochs: int = 1000,
    batches_per_epoch: int = 2**9,
    embedding_size: int = 100,
    holdouts_number: int = 10,
    thresholds: Tuple[int] = (
        None, 10, 100, 1000, 2000, 3000, 4000, 5000, 6000)
) -> pd.DataFrame:
    results = []
    for holdout_number, pos_train, pos_test, neg_train, neg_test in load_complete_string_ppi(
        holdouts_number=holdouts_number
    ):
        for threshold in tqdm(thresholds, desc="Thresholds"):
            embedding = compute_skipgram_embedding(
                graph=pos_train,
                holdout=holdout_number,
                root=root,
                embedding_size=embedding_size,
                epochs=epochs,
                max_neighbours=threshold
            )
            transformer = LinkPredictionTransformer(method="Hadamard")
            transformer.fit(embedding)
            x_train, y_train = transformer.transform(
                pos_train,
                neg_train,
                aligned_node_mapping=True
            )
            x_test, y_test = transformer.transform(
                pos_test,
                neg_test,
                aligned_node_mapping=True
            )
            train_pred, test_pred = get_perceptron_predictions(
                graph=pos_train,
                max_neighbours=threshold,
                holdout=holdout_number,
                root=root,
                embedding=embedding,
                epochs=epochs,
                batches_per_epoch=batches_per_epoch,
                x_train=x_train,
                x_test=x_test
            )
            results.append({
                "run_type": "train",
                "link_prediction_model": "Perceptron",
                "embedding_model": "SkipGram",
                "Max Neighbours": threshold,
                **compute_metrics(y_train, train_pred),
            })
            results.append({
                "run_type": "test",
                "link_prediction_model": "Perceptron",
                "embedding_model": "SkipGram",
                "Max Neighbours": threshold,
                **compute_metrics(y_test, test_pred),
            })
    return pd.DataFrame(results)
