"""Module providing APIs towards SNAP Node2Vec."""
import pandas as pd
from .abstract_graph_embedding_library import AbstractGraphEmbeddingLibrary
import numpy as np
from multiprocessing import cpu_count

class FastNode2VecLibrary(AbstractGraphEmbeddingLibrary):

    @staticmethod
    def get_library_name() -> str:
        """Returns the name of the library."""
        return "FastNodeEmbedding"

    @staticmethod
    def store_graph(graph, edge_list_path: str):
        """Store the provided graph to the provided path in the current library format.

        Parameters
        -------------------------
        graph: Graph
            The graph to be made available for SNAP.
        edge_list_path: str
            The path where to store the graph.
        """
        np.save(f"{edge_list_path}.npy", graph.get_edge_node_ids(directed=False))

    def compute_node_embedding(
        self,
        edge_list_path: str,
        embedding_path: str,
        embedding_size: int,
        random_walk_length: int,
        iterations_per_node: int,
        epochs: int,
        p: float,
        q: float,
        window_size: int
    ):
        """Compute node embedding using SNAP's Node2Vec.

        Parameters
        -------------------------
        path: str
            Path from where to load the graph
        embedding_path: str
            Path where to store the embedding
        embedding_size: int
            Size of the embedding
        random_walk_length: int
            Length of the random walk
        iterations_per_node: int
            Number of iterations to execute per node
        epochs: int
            Number of epochs to run the embedding for
        p: float
            Value of the explore weight
        q: float
            Value of the return weight
        window_size: int
            Size of the context.
        """
        from fastnode2vec import Graph, Node2Vec

        graph = Graph(
            np.load(f"{edge_list_path}.npy"),
            directed=False,
            weighted=False
        )
        n2v = Node2Vec(
            graph,
            dim=embedding_size,
            walk_length=random_walk_length,
            context=window_size,
            p=p,
            q=q,
            batch_walks=iterations_per_node,
            workers=cpu_count()
        )
        # We are using the same number of epochs
        # as the other models, but as it is not possible
        # to specify the number of 
        n2v.train(epochs=epochs*iterations_per_node)

        n2v.wv[graph.node_names]
        pd.DataFrame(
            n2v.wv[graph.node_names],
            index=graph.node_names
        ).to_csv(embedding_path)

    @staticmethod
    def load_embedding(
        graph,
        embedding_path: str,
    ) -> pd.DataFrame:
        """Returns embedding computed from the SNAP Node2Vec implementation.

        Parameters
        --------------------------
        graph: Graph
            The graph associated to the embedding.
        embedding_path: str
            The path from where to load the embedding.
        """
        # Load the Embedding from the provided path.
        embedding = pd.read_csv(
            embedding_path,
            index_col=0
        )
        
        # The nodes should be a numeric range,
        # but may not be sorted.
        embedding = embedding.loc[graph.get_node_ids()]

        # Reindex it to make sure it is aligned with provided graph.
        embedding.index = graph.get_node_names()

        return embedding