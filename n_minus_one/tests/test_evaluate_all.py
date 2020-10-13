from src import evaluate_all


def test_evaluate_all():
    """Test if the complete pipeline runs."""
    evaluate_all(
        "tests/test_parameters.json",
        "data/macaque.tsv",
        graph_name="Macaque",
        has_weights=False,
        mlp_epochs=1,
        embedder_epochs=1
    )
