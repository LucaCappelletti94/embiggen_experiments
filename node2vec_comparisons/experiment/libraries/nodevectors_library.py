"""Module providing APIs towards SNAP Node2Vec."""
import pandas as pd
from .abstract_graph_embedding_library import AbstractGraphEmbeddingLibrary


class NodeVectorsLibrary(AbstractGraphEmbeddingLibrary):

    @staticmethod
    def get_library_name() -> str:
        """Returns the name of the library."""
        return "NodeVectors"

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
        pd.DataFrame(
            graph.get_directed_edge_node_ids()
        ).to_csv(
            edge_list_path,
            sep="\t",
            index=False,
            header=False
        )

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
        import csrgraph as cg
        from nodevectors import Node2Vec

        G = cg.read_edgelist(
            edge_list_path,
            directed=False,
            header=None,
            sep='\t'
        )
        g2v = Node2Vec(
            n_components=embedding_size,
            walklen=random_walk_length,
            return_weight=p,
            neighbor_weight=q,
            w2vparams=dict(
                window=window_size,
                epochs=epochs
            )
        )
        embedding = g2v.fit_transform(G)
        pd.DataFrame(
            embedding,
            index=G.names
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
            embedding_path
        )
        # Reindex it to make sure it is aligned with provided graph.
        return embedding.loc[graph.get_node_names()]
