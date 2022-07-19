import numpy as np
from autoencoder_model import run_experiments, process_experiments, set_seeds
from paths import MNIST_EXPERIMENT_PATH
from tensorflow.keras.datasets import mnist

'''
Simply replicating experiment from: https://blog.keras.io/building-autoencoders-in-keras.html
'''

def run_mnist(seed):
    set_seeds(seed)
    run_type = "no_batching"
    (x_train,_), (x_test,_) = mnist.load_data()
    # Normalise and flatten
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    run_histories = run_experiments(x_train, x_test, run_type=run_type, experiment_path=MNIST_EXPERIMENT_PATH)
    process_experiments(name="mnist", experiment_path=MNIST_EXPERIMENT_PATH)
    return run_histories

if __name__ == "__main__":
    run_mnist(seed = 1)
