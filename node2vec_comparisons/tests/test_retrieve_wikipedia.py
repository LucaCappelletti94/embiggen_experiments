"""Submodule to test the retrieval of the wikipedia file."""
from experiment.data_retrieval import retrieve_coo_wikipedia


def test_retrieve_coo_wikipedia():
    """Unit test to verify that COO retrieval pipeline works."""
    graph = retrieve_coo_wikipedia()
    print(
        "Nodes number: ", graph.get_nodes_number(),
        "Edges number: ", graph.get_edges_number(),
        "Singletons: ", graph.get_singleton_nodes_number() 
    )