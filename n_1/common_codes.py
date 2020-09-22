import silence_tensorflow.auto # Import needed to avoid TensorFlow warnings and general useless infos.
from ensmallen_graph import EnsmallenGraph
from embiggen import Node2VecSequence

from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.optimizers import Nadam
from embiggen import SkipGram
from embiggen import CBOW
from embiggen import GloVe
from tensorflow.keras.callbacks import EarlyStopping
from plot_keras_history import plot_history
import numpy as np
import matplotlib.pyplot as plt


class GrapEmbedding:
    def __init__(self, edge_path, walk_length, batch_size, iterations, window_size, p, q, delta, patience, embedding_size, negative_samples
                 , embedding_model, learning_rate,epochs,glove_alpha):
        self.edgelist_path = edge_path
        self.walk_length = walk_length
        self.batch_size = batch_size
        self.iterations = iterations
        self.window_size = window_size
        self.p = p
        self.q = q
        self.delta = delta
        self.patience = patience
        self.embedding_size = embedding_size
        self.negative_samples = negative_samples
        self.embedding_model = embedding_model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.glove_alpha = glove_alpha



    def read_graph(self):
        graph = EnsmallenGraph.from_csv(
        edge_path=self.edgelist_path, #data/ppi/edges.tsv
        sources_column="subject",
        destinations_column="object",
        directed=False,
        weights_column="weight"
        )
        print(graph.report())
        return graph

    def create_training_validation(self,graph):
        training, validation = graph.connected_holdout(42, 0.8)
        assert graph > training
        assert graph > validation
        assert (training + validation).contains(graph)
        assert graph.contains(training + validation)
        assert not training.overlaps(validation)
        assert not validation.overlaps(training)
        return training, validation



    def train_seq(self,training):
        training_sequence = Node2VecSequence(
        training,
        walk_length=self.walk_length,
        batch_size=self.batch_size,
        iterations=self.iterations,
        window_size=self.window_size,
        return_weight=1/self.p,
        explore_weight=1/self.q
        )
        return training_sequence

    def valid_seq(self,graph):
        validation_sequence = Node2VecSequence(
        graph, # Here we use the entire graph. This will only be used for the early stopping.
        walk_length=self.walk_length,
        batch_size=self.batch_size,
        iterations=self.iterations,
        window_size=self.window_size,
        return_weight=1/self.p,
        explore_weight=1/self.q
        )
        return validation_sequence



    def train_glove(self, training):

        train_words, train_contexts, train_labels = training.cooccurence_matrix(
            walk_length=self.walk_length,
            window_size=self.window_size,
            iterations=self.iterations,
            return_weight=1 / self.p,
            explore_weight=1 / self.q
        )
        return train_words, train_contexts, train_labels

    def valid_glove(self,graph):
        valid_words, valid_contexts, valid_labels = graph.cooccurence_matrix(
        walk_length=self.walk_length,
        window_size=self.window_size,
        iterations=self.iterations,
        return_weight=1 / self.p,
        explore_weight=1 / self.q
        )
        return valid_words, valid_contexts, valid_labels


    def embed_model(self,  training):
        strategy = MirroredStrategy()
        if self.embedding_model == "skipgram":
             with strategy.scope():
                 model = SkipGram(
                     vocabulary_size=training.get_nodes_number(),
                     embedding_size=self.embedding_size,
                     window_size=self.window_size,
                     negatives_samples=self.negative_samples,
                     optimizer=Nadam(learning_rate=self.learning_rate)
                 )
        elif self.embedding_model == "cbow":
            with strategy.scope():
                model = CBOW(
                    vocabulary_size=training.get_nodes_number(),
                    embedding_size=self.embedding_size,
                    window_size=self.window_size,
                    negatives_samples=self.negative_samples,
                    optimizer=Nadam(learning_rate=self.learning_rate)
                )
        elif self.embedding_model == "glove":
            with strategy.scope():
                model = GloVe(
                    vocabulary_size=training.get_nodes_number(),
                    embedding_size=self.embedding_size,
                    alpha=self.glove_alpha
                )

        print(model.summary())
        return model

    def history(self,training_sequence,validation_sequence,model):
        history = model.fit(
        training_sequence,
        steps_per_epoch=training_sequence.steps_per_epoch,
        validation_data=validation_sequence,
        validation_steps=validation_sequence.steps_per_epoch,
        epochs=self.epochs,
        callbacks=[
            EarlyStopping(
                "val_loss",
                min_delta=self.delta,
                patience=self.patience,
                restore_best_weights=True
            )
        ]
        )

        model.save_weights(f"{model.name}_weights.h5")
        plot_history(history)
        plt.savefig("history.png")
        np.save(f"{model.name}_embedding.npy", model.embedding)


    def history_glove(self, model, train_words, train_contexts,train_labels,valid_words,valid_contexts,valid_labels):
        history = model.fit(
            (train_words, train_contexts), train_labels,
            validation_data=((valid_words, valid_contexts), valid_labels),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[
                EarlyStopping(
                    "val_loss",
                    min_delta=self.delta,
                    patience=self.patience,
                    restore_best_weights=True
                )
            ]
        )
        model.save_weights(f"{model.name}_weights.h5")
        plot_history(history)
        plt.savefig("history.png")
        np.save(f"{model.name}_embedding.npy", model.embedding)