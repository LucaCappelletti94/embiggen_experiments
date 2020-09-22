import silence_tensorflow.auto # Import needed to avoid TensorFlow warnings and general useless infos.
from ensmallen_graph import EnsmallenGraph
from embiggen import Node2VecSequence
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.optimizers import Nadam
from embiggen import CBOW
from tensorflow.keras.callbacks import EarlyStopping
from plot_keras_history import plot_history
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
embedding_model = "cbow"


graph_embed = GrapEmbedding(edge_path, walk_length, batch_size, iterations, window_size, p, q, delta, patience, embedding_size,
                            negative_samples, embedding_model, learning_rate,epochs)

graph = graph_embed.read_graph()
training,validation = graph_embed.create_training_validation(graph)
training_sequence = graph_embed.train_seq(training)
validation_sequence = graph_embed.valid_seq(graph)
model = graph_embed.embed_model(training)
history = graph_embed.history(training_sequence, validation_sequence, model)