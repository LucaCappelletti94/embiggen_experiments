"""Module providing abstract graph embedding library class."""
from ensmallen import Graph


class AbstractGraphEmbeddingLibrary:

    @staticmethod
    def get_library_name() -> str:
        """Returns the name of the library."""
        raise NotImplementedError(
            "The method `get_library_name` should "
            "be implemented in the child classes."
        )

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
        raise NotImplementedError(
            "The method store graph should "
            "be implemented in the child classes."
        )

    def _load_graph(self, edge_list_path: str):
        """Return graph in the format for the current graph library.
        
        Parameters
        --------------------
        edge_list_path: str
            The path from where to load the graph.
        """
        raise NotImplementedError(
            "The method load graph should "
            "be implemented in the child classes."
        )