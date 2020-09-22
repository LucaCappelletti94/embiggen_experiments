import silence_tensorflow.auto # Import needed to avoid TensorFlow warnings and general useless infos.
from ensmallen_graph import EnsmallenGraph
from tensorflow.distribute import MirroredStrategy
from embiggen import GloVe
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as mpl

from common_codes import GrapEmbedding

edge_path="data/ppi/edges.tsv" #modify if needed
#get_parameters()#TODO read parameters from a json file
walk_length=100
batch_size=2**7
iterations=20
window_size=4
p=1.0
q=1.0
embedding_size=100
negative_samples=30
patience=5
delta=0.0001
epochs=1000
learning_rate=0.1
embedding_model = "glove"
glove_alpha = 0.75


graph_embed = GrapEmbedding(edge_path, walk_length, batch_size, iterations, window_size, p, q, delta, patience, embedding_size,
                            negative_samples, embedding_model, learning_rate,epochs,glove_alpha)

graph = graph_embed.read_graph()
training,validation = graph_embed.create_training_validation(graph)
train_words, train_contexts, train_labels = graph_embed.train_glove(training)
valid_words, valid_contexts, valid_labels = graph_embed.valid_glove(graph)
model = graph_embed.embed_model(training)
history = graph_embed.history_glove(model, train_words, train_contexts,train_labels,valid_words,valid_contexts,valid_labels)












