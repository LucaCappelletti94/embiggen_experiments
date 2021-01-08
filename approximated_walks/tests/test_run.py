from experiment import run


def test_run():
    """Test that everything works in run."""
    run(
        root="test_results",
        epochs=1,
        batches_per_epoch=1,
        embedding_size=1,
        holdouts_number=1,
        thresholds=1
    )
