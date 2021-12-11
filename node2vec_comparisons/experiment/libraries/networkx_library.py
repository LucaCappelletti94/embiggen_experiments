"""Module providing APIs towards SNAP Node2Vec."""
from ensmallen import Graph
import pandas as pd
import networkx as nx
from .abstract_graph_embedding_library import AbstractGraphEmbeddingLibrary


class NetworkXLibrary(AbstractGraphEmbeddingLibrary):

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

    def _load_graph(self, edge_list_path: str) -> nx.Graph:
        """Return graph in the format for the current graph library.

        Parameters
        --------------------
        edge_list_path: str
            The path from where to load the graph.
        """
        return nx.read_edgelist(
            edge_list_path,
            data=False,
            delimiter=" "
        )
