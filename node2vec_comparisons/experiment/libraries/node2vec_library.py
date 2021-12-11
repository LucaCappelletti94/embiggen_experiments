"""Module providing APIs towards the Graph Embedding library."""
from ensmallen import Graph
import pandas as pd
from node2vec import Node2Vec
from multiprocessing import cpu_count
from .networkx_library import NetworkXLibrary


class Node2VecLibrary(NetworkXLibrary):

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
        """Compute node embedding using Node2Vec.

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
        graph = self._load_graph(edge_list_path)
        model = Node2Vec(
            graph,
            dimensions=embedding_size,
            walk_length=random_walk_length,
            num_walks=iterations_per_node,
            p=p,
            q=q,
            workers=cpu_count()
        )
        model.fit(
            window=window_size,
            epochs=epochs
        )
        pd.DataFrame(model.get_embeddings()).T.to_csv(embedding_path)

    def load_embedding(
        graph: Graph,
        embedding_path: str,
    ) -> pd.DataFrame:
        """Returns embedding.

        Parameters
        --------------------------
        graph: Graph
            The graph associated to the embedding.
        embedding_path: str
            The path from where to load the embedding.
        """
        # Load the Embedding from the provided path.
        embedding = pd.read_csv(
            embedding_path
        )
        # Reindex it to make sure it is aligned with provided graph.
        return embedding.loc[graph.get_node_names()]