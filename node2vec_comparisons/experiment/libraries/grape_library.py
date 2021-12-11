"""Module providing APIs towards SNAP Node2Vec."""
from ensmallen import Graph
from embiggen import GraphCBOW
import pandas as pd
from .abstract_graph_embedding_library import AbstractGraphEmbeddingLibrary
import compress_json


class GraPELibrary(AbstractGraphEmbeddingLibrary):

    @staticmethod
    def get_library_name() -> str:
        """Returns the name of the library."""
        return "GraPE"

    @staticmethod
    def store_graph(graph: Graph, edge_list_path: str):
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
                edges_number=graph.get_directed_edges_number()
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

        model = GraphCBOW(
            graph,
            embedding_size=embedding_size,
            walk_length=random_walk_length,
            iterations=iterations_per_node,
            window_size=window_size,
            return_weight=1/p,
            explore_weight=1/q,
            max_neighbours=10_000,
            batch_size=64
        )

        model.fit(
            epochs=epochs
        )

    def load_embedding(
        graph: Graph,
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