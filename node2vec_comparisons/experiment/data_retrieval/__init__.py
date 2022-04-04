"""Submodule providing methods to retrieve the data used within the experiments."""
from .ctd import retrieve_ctd
from .wikipedia import retrieve_english_wikipedia
from .retrieve_graphs import retrieve_coo_ctd, retrieve_coo_pheknowlator, retrieve_coo_wikipedia

__all__ = [
    "retrieve_ctd",
    "retrieve_english_wikipedia",
    "retrieve_coo_ctd",
    "retrieve_coo_pheknowlator",
    "retrieve_coo_wikipedia"
]
