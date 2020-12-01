from glob import glob
import pandas as pd
from experiment import run

if __name__ == "__main__":
    for path in glob("data/*/*.tsv*.xz"):
        pd.read_csv(path, sep="\t").to_csv(path[:-3], sep="\t", index=False)
    run().to_csv("yue_comparisons.csv", index=False)
