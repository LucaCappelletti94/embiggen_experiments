# Node2Vec comparisons
In this experiment, we compare the memory and time performance of different libraries providing Node2Vec implementations. Furthermore, we compare the performance of a simple link prediction model trained on the aforementioned node embeddings.

## Considered libraries
We are comparing the following libraries:

* [SNAP](https://github.com/snap-stanford/snap/tree/master/examples/node2vec)
* [NodeVectors](https://github.com/VHRanger/nodevectors), specifically [from our fork](https://github.com/LucaCappelletti94/nodevectors).
* [GraPE/Embiggen](https://github.com/monarch-initiative/embiggen)
* [GraphEmbedding](https://github.com/shenweichen/GraphEmbedding), specifically [from our fork](https://github.com/LucaCappelletti94/GraphEmbedding).
* [Node2Vec](https://github.com/eliorc/node2vec), specifically [from our fork](https://github.com/LucaCappelletti94/node2vec).

Note that we created forks of the libraries exclusively in order to make them compatible with the updated version of their dependencies.

### Yet not integrated libraries
We'd like to add [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric) to the benchmark, but so far we were not able to properly install and execute the library in our envinronment to run its version of Node2Vec.

## Considered datasets
We have taking into consideration three datasets, namely [CTD](http://ctdbase.org/), [PheKnowLator](https://github.com/callahantiff/PheKnowLator) and [English Wikipedia](https://dumps.wikimedia.org/backup-index.html).

We provide in the [`data_retrieval`](https://github.com/LucaCappelletti94/embiggen_experiments/tree/node2vec_comparisons/node2vec_comparisons/experiment/data_retrieval) submodule the scripts to automatically retrieve and build the CTD and English Wikipedia graphs. The PheKnowLator graph may be retrieved from the [`PheKnowLator Gcloud directory`](https://console.cloud.google.com/storage/browser/pheknowlator/archived_builds/release_v3.0.2/build_18OCT2021/knowledge_graphs/subclass_builds/inverse_relations/owlnets;tab=objects?prefix=&forceOnObjectsSortingFiltering=false).

Since running the scripts to process the graphs is both time consuming and has a relatively high memory peak, we provide the processed edge lists from [Internet Archive](https://archive.org/), which we retrieve automatically at the beginning of the experiments:

* [Wikipedia edge list](https://archive.org/details/wikipedia_edge_list.npy)
* [CTD](https://archive.org/details/ctd_20220404)
* [PheKnowLator](https://archive.org/details/phe-know-lator)

To automatically retrieve and load the three graphs you can use the following:

```python
from experiment.data_retrieval import retrieve_coo_ctd, retrieve_coo_pheknowlator, retrieve_coo_wikipedia

ctd = retrieve_coo_ctd()
pheknowlator = retrieve_coo_pheknowlator()
wikipedia = retrieve_coo_wikipedia()
```

Specifically, the graph versions we have taken in consideration have the following sizes:

| Graph name        | Number of nodes | Number of edges |
|-------------------|-----------------|-----------------|
| CTD               | 104008          | 45107392        |
| PheKnowLator      | 780375          | 7442356         |
| English Wikipedia | 17342950        | 130515014       |

## Running the experiments
To run the experiments, first install the dependecies specified in the `requirements.txt` document present in this
directory by running:

```bash
pip install -r requirements.txt
```

Secondly, to run experiments, from the repository directory run:

Within the gcloud system with 64 CPUs and 64GBs of RAM, run the PecanPy benchmark:

```bash
python3 run_pecanpy.py embedding
```

Always within the gcloud system, on a machine with 4 GPUs with **equivalen cost of the computation time** as the previous machine run:

```bash
python3 run_grape.py embedding
```

On a machine where you intend to run the edge prediction run:

```bash
python3 run_pecanpy.py edge_prediction
python3 run_grape.py edge_prediction
```