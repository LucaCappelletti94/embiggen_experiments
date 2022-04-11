"""Submodule to test the retrieval of the CTD file."""
from experiment.data_retrieval import retrieve_coo_ctd


def test_retrieve_ctd():
    """Unit test to verify that COO retrieval pipeline works."""
    graph = retrieve_coo_ctd()
    print(graph.get_unique_node_type_names())
    print(
        "Nodes number: ", graph.get_nodes_number(),
        "Edges number: ", graph.get_edges_number(),
        "Singletons: ", graph.get_singleton_nodes_number() 
    )
