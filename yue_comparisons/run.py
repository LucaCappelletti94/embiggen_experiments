from glob import glob
import pandas as pd
from experiment import run

if __name__ == "__main__":
    run().to_csv("yue_comparisons.csv", index=False)
