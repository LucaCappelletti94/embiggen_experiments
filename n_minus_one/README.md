# N minus one

To execute the experiment, run the following:

```python
from src import evaluate_all


evaluate_all(
    embedding_model="GloVe",
    results_folder="results/",
    parameters_path="tests/test_parameters.json",
    graph_path="data/macaque.tsv",
    graph_name="Macaque",
    has_weights=False,
    mlp_epochs=1,
    embedder_epochs=10
)

```