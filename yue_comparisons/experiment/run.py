from embiggen import LinkPredictionTransformer
import pandas as pd
from .link_prediction import get_ffnn_predictions, get_perceptron_predictions, load_link_prediction_graphs
from .embedding import compute_cbow_embedding, compute_glove_embedding, compute_skipgram_embedding
from .utils import compute_metrics


def run(
    root: str = "./",
    epochs: int = 1000,
    batches_per_epoch: int = 2**9,
    embedding_size: int = 100,
    holdouts_number: int = 10
) -> pd.DataFrame:
    embedding_models = [
        compute_cbow_embedding,
        compute_glove_embedding,
        compute_skipgram_embedding
    ]
    link_prediction_models = [
        get_ffnn_predictions,
        get_perceptron_predictions
    ]
    results = []
    for holdout_number, graph_name, pos_train, pos_test, neg_train, neg_test in load_link_prediction_graphs(
        holdouts_number=holdouts_number
    ):
        for embedding_model in embedding_models:
            embedding = embedding_model(
                graph=pos_train,
                graph_name=graph_name,
                holdout=holdout_number,
                root=root,
                embedding_size=embedding_size,
                epochs=epochs
            )
            print(embedding)
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
            for link_prediction_model in link_prediction_models:
                train_pred, test_pred = link_prediction_model(
                    graph=pos_train,
                    graph_name=pos_train.get_name(),
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
                    **compute_metrics(y_train, train_pred),
                })
                results.append({
                    "run_type": "test",
                    **compute_metrics(y_test, test_pred),
                })
    return pd.DataFrame(results)
