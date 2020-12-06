from experiment import run
import tensorflow as tf

if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        run().to_csv("approximated_walks.csv", index=False)