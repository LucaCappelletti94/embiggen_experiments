"""Submodule to test the retrieval of the PheKnowLator file."""
from experiment.data_retrieval import retrieve_coo_pheknowlator


def test_retrieve_coo_pheknowlator():
    """Unit test to verify that COO retrieval pipeline works."""
    graph = retrieve_coo_pheknowlator()
    print(
        "Nodes number: ", graph.get_nodes_number(),
        "Edges number: ", graph.get_edges_number(),
        "Singletons: ", graph.get_singleton_nodes_number() 
    )