import numpy as np
from tensorflow.keras.datasets import mnist

from src.autoencoder_model import run_experiments, process_experiments, set_seeds
from src.paths import MNIST_EXPERIMENT_PATH


def run_mnist(seed, num_epochs, lr, middle_node_size, sample_size=1):
    """
    Simply replicating experiment from: https://blog.keras.io/building-autoencoders-in-keras.html
    """
    set_seeds(seed)
    run_type = "no_batching"
    (x_train, _), (x_test, _) = mnist.load_data()
    # Normalise and flatten
    IMAGE_NORM_CONST = 255.0
    x_train = x_train.astype("float32") / IMAGE_NORM_CONST
    x_test = x_test.astype("float32") / IMAGE_NORM_CONST
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    all_data = np.concatenate((x_train, x_test))

    """We use the standard split from keras if sample_size >= 1"""
    if sample_size < 1:
        amount_of_data = int(all_data.shape[0] * sample_size)
        training_indices = np.random.choice(
            all_data.shape[0], amount_of_data, replace=False
        )
        x_train = all_data[training_indices]
        test_mask = np.ones(len(all_data), np.bool)
        test_mask[training_indices] = 0
        x_test = all_data[test_mask]

    run_histories = run_experiments(
        x_train,
        x_test,
        run_type=run_type,
        experiment_path=MNIST_EXPERIMENT_PATH,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=256,
        middle_node_size=middle_node_size,
    )
    process_experiments(name="mnist", experiment_path=MNIST_EXPERIMENT_PATH)

    return run_histories


if __name__ == "__main__":
    run_mnist(1, 10, 0.1, 32)
