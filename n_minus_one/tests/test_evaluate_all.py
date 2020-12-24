import os
from src import evaluate_all
import shutil


def test_evaluate_all():
    """Test if the complete pipeline runs."""
    root = "tests/results"
    if os.path.exists(root):
        shutil.rmtree(root)
    evaluate_all(
        root,
        "tests/test_parameters.json",
        "data/macaque.tsv",
        graph_name="Macaque",
        has_weights=False,
        mlp_epochs=1,
        embedder_epochs=10
    )
    shutil.rmtree(root)
