from experiment import run
from tensorflow.distribute import MirroredStrategy

if __name__ == "__main__":
    with MirroredStrategy().scope():
        run().to_csv("approximated_walks.csv", index=False)