import silence_tensorflow.auto # Import needed to avoid TensorFlow warnings and general useless infos.
import pandas as pd
import numpy as np
from embiggen import CorpusTransformer
from embiggen import Word2VecSequence
from tensorflow.distribute import MirroredStrategy
from embiggen import CBOW
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from plot_keras_history import plot_history
import matplotlib.pyplot as plt
import os



batch_size = 16
embedding_size = 100
window_size=5
negative_samples=20
delta=0.0001
patience=20
min_counts=5


pubmed = pd.read_csv("/data/small_pubmed_cr.tsv",header = None,sep='\t').iloc[:,2].tolist()



transformer = CorpusTransformer()
transformer.fit(pubmed, min_count=min_counts)
encoded_pubmed = transformer.transform(pubmed, min_length=window_size*2+1)
np.save(f"encoded_pubmed.npy",encoded_pubmed)

print(f"The transformer will use {transformer.vocabulary_size} different terms.")

word2vec_sequence = Word2VecSequence(
    encoded_pubmed,
    batch_size=batch_size,
    window_size=window_size,
    support_mirror_strategy = True
)


strategy = MirroredStrategy()
with strategy.scope():
    model = CBOW(
        vocabulary_size=transformer.vocabulary_size,
    	embedding_size=embedding_size,
    	optimizer=Nadam(0.01),
    	window_size=window_size,
    	negative_samples=negative_samples,
	)

print(model.summary())

history = model.fit(
    word2vec_sequence,
    steps_per_epoch=word2vec_sequence.steps_per_epoch,
    epochs=1000,
    callbacks=[
        EarlyStopping(
            "loss",
            min_delta=delta,
            patience=patience,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(monitor="loss", patience=patience//2)
    ]
)
print(len(model.embedding))


np.save(f"embedding.npy", model.embedding)

with open("words.txt", "w") as f_write:
    for i in range(1,len(model.embedding)+1):
        f_write.write("{}\n".format(transformer.reverse_transform([[i]])))


plot_history(history)
plt.savefig("history.png")
