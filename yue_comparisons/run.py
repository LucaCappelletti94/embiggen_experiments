import setGPU
import sys
from glob import glob
import pandas as pd
from experiment import run
from experiment.embedding import compute_cbow_embedding, compute_glove_embedding, compute_skipgram_embedding

if __name__ == "__main__":
    model = sys.argv[1]
    models = {
        "cbow": compute_cbow_embedding,
        "glove": compute_glove_embedding,
        "skipgram": compute_skipgram_embedding
    }
    if model in models:
        raise ValueError(
            "The model {} is not available.".format(model)
        )
    run(models[model]).to_csv("yue_comparisons.csv", index=False)
