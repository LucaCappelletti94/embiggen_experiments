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

## Yet not integrated libraries
We'd like to add [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric) to the benchmark, but so far we were not able to install the library.