from experiment import run
from tensorflow.distribute import MirrorStrategy

if __name__ == "__main__":
    with MirrorStrategy().scope():
        run().to_csv("approximated_walks.csv", index=False)