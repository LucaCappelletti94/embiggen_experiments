from experiment import run


def test_run():
    run(
        root="test_run",
        epochs=1,
        batches_per_epoch=1,
        embedding_size=1,
        holdouts_number=1
    )
