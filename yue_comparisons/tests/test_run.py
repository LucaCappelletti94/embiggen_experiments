from experiment import run
from experiment.embedding import compute_cbow_embedding, compute_glove_embedding, compute_skipgram_embedding


def test_run():
    for embedding_model in (compute_cbow_embedding, compute_glove_embedding, compute_skipgram_embedding):
        run(
            embedding_model,
            root="test_run",
            epochs=1,
            batches_per_epoch=1,
            embedding_size=1,
            holdouts_number=1,
            verbose=True
        )
