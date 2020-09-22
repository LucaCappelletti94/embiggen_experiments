import silence_tensorflow.auto # Import needed to avoid TensorFlow warnings and general useless infos.
from ensmallen_graph import EnsmallenGraph
from tensorflow.distribute import MirroredStrategy
from embiggen import GloVe
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as mpl


graph = EnsmallenGraph.from_csv(
    edge_path="data/ggi/edges.tsv",
    sources_column="subject",
    destinations_column="object",
    directed=False,
    weights_column="weight"
)

graph.report()
training, validation = graph.connected_holdout(42, 0.8)
assert graph > training
assert graph > validation
assert (training + validation).contains(graph)
assert graph.contains(training + validation)
assert not training.overlaps(validation)
assert not validation.overlaps(training)

walk_length=100
batch_size=2**20
iterations=20
window_size=4
p=1.0
q=1.0
embedding_size=100
patience=5
delta=0.0001
epochs=1000
learning_rate=0.1
glove_alpha=0.75

train_words, train_contexts, train_labels = training.cooccurence_matrix(
    walk_length=walk_length,
    window_size=window_size,
    iterations=iterations,
    return_weight=1/p,
    explore_weight=1/q
)

valid_words, valid_contexts, valid_labels = graph.cooccurence_matrix(
    walk_length=walk_length,
    window_size=window_size,
    iterations=iterations,
    return_weight=1/p,
    explore_weight=1/q
)

strategy = MirroredStrategy()
with strategy.scope():
    model = GloVe(
        vocabulary_size=training.get_nodes_number(),
        embedding_size=embedding_size,
        alpha=glove_alpha
    )

model.summary()

history = model.fit(
    (train_words, train_contexts), train_labels,
    validation_data=((valid_words, valid_contexts), valid_labels),
    epochs=1000,
    batch_size=batch_size,
    callbacks=[
        EarlyStopping(
            "val_loss",
            min_delta=delta,
            patience=patience,
            restore_best_weights=True
        )
    ]
)
model.save_weights(f"{model.name}_weights.h5")

from plot_keras_history import plot_history

plot_history(history)

mpl.savefig("history.png")


np.save(f"{model.name}_embedding.npy", model.embedding)


