"""Module providing APIs towards SNAP Node2Vec."""
import pandas as pd
from .abstract_graph_embedding_library import AbstractGraphEmbeddingLibrary
import compress_json


class GraPECPULibrary(AbstractGraphEmbeddingLibrary):

    @staticmethod
    def get_library_name() -> str:
        """Returns the name of the library."""
        return "GraPECPU"

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
            sep=" ",
            index=False,
            header=False
        )
        compress_json.dump(
            dict(
                nodes_number=graph.get_nodes_number(),
                edges_number=graph.get_number_of_directed_edges()
            ),
            "{}.metadata.json".format(edge_list_path)
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
        edge_list_path: str
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
        from ensmallen import Graph
        from embiggen import GraphCBOW

        graph = Graph.from_csv(
            edge_path=edge_list_path,
            directed=False,
            **compress_json.load("{}.metadata.json".format(edge_list_path)),
            sources_column_number=0,
            destinations_column_number=1,
            numeric_node_ids=True,
            edge_list_numeric_node_ids=True,
            verbose=False,
            edge_list_header=False,
            edge_list_is_complete=True,
            edge_list_may_contain_duplicates=False,
            edge_list_is_sorted=True,
            edge_list_is_correct=True,
            load_edge_list_in_parallel=True
        )

        graph.enable()

        pd.DataFrame(
            graph.compute_cbow_embedding(
                embedding_size=embedding_size,
                iterations=iterations_per_node,
                walk_length=random_walk_length,
                window_size=window_size,
                epochs=epochs,
                return_weight=1/p,
                explore_weight=1/q,
                max_neighbours=100,
            ),
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

        # The nodes should be a numeric range,
        # but may not be sorted.
        embedding = embedding.loc[graph.get_node_ids()]

        # Reindex it to make sure it is aligned with provided graph.
        embedding.index = graph.get_node_names()

        return embedding
