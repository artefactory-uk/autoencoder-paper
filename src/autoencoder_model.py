import pandas as pd
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.losses import mean_squared_error
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LambdaCallback
import pathlib


FULL_BATCH_SIZE = 9999999
FIRST_LAYER_SIZE = 64

INITIALISER_DICT = {
    "straddled": "straddled",
    "glorotuniform": tf.keras.initializers.GlorotUniform(),
    "glorotnormal": tf.keras.initializers.GlorotNormal(),
    "identity": tf.keras.initializers.Identity(),
    "henormal": tf.keras.initializers.HeNormal(),
    "heuniform": tf.keras.initializers.HeUniform(),
    "orthogonal": tf.keras.initializers.Orthogonal(),
    "random": tf.keras.initializers.RandomNormal(),
}


def root_mean_squared_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return K.sqrt(mean_squared_error(y_true, y_pred))


def set_seeds(x: int):
    np.random.seed(x)
    tf.random.set_seed(x)


def recurrent_identity_matrix(shape1: int, shape2: int) -> np.ndarray:
    if shape1 == shape2:
        return np.identity(shape1)
    elif shape1 > shape2:
        matrix_without_zeroes = np.concatenate(
            [np.identity(shape2)] * (shape1 // shape2), axis=0
        )
        return np.concatenate(
            (matrix_without_zeroes, np.zeros((shape1 % shape2, shape2))), axis=0
        )
    else:
        matrix_without_zeroes = np.concatenate(
            [np.identity(shape1)] * (shape2 // shape1), axis=1
        )
        return np.concatenate(
            (matrix_without_zeroes, np.zeros((shape1, shape2 % shape1))), axis=1
        )


def straddled_matrix(
    shape1: int, shape2: int, add_glorot: bool = False, symmetric: bool = False
) -> np.ndarray:
    initializer = tf.keras.initializers.GlorotUniform()
    glorot_uniform = initializer(shape=(shape1, shape2))

    def tall_straddled(shape1: int, shape2: int) -> np.ndarray:
        small_matrix = np.identity(shape2)
        matrix = small_matrix
        for i in range(ceil(shape1 / shape2)):
            matrix = np.concatenate((matrix, small_matrix), axis=0)
        return matrix[:shape1, :]

    if symmetric:
        if shape1 >= shape2:
            straddled = tall_straddled(shape1, shape2)
        else:
            straddled = tall_straddled(shape2, shape1).T
    else:
        straddled = tall_straddled(shape1, shape2)

    if add_glorot:
        straddled += glorot_uniform
        straddled = straddled.numpy()

    return straddled


class AnomalyDetector(tf.keras.Model):
    def __init__(
        self,
        first_layer_size: int,
        no_of_features: int,
        middle_layer_size: int,
        initialiser_key: str = "straddled",
    ):
        super().__init__()
        if initialiser_key == "straddled":

            self.encoder = tf.keras.Sequential(
                [
                    layers.Dense(
                        first_layer_size,
                        activation="relu",
                        use_bias=True,
                        kernel_initializer=tf.constant_initializer(
                            straddled_matrix(no_of_features, first_layer_size)
                        ),
                    ),
                    layers.Dense(
                        middle_layer_size,
                        activation="relu",
                        use_bias=True,
                        kernel_initializer=tf.constant_initializer(
                            straddled_matrix(first_layer_size, middle_layer_size)
                        ),
                    ),
                ]
            )

            self.decoder = tf.keras.Sequential(
                [
                    layers.Dense(
                        first_layer_size,
                        activation="relu",
                        use_bias=True,
                        kernel_initializer=tf.constant_initializer(
                            straddled_matrix(middle_layer_size, first_layer_size)
                        ),
                    ),
                    layers.Dense(
                        no_of_features,
                        activation="sigmoid",
                        use_bias=True,
                        kernel_initializer=tf.constant_initializer(
                            straddled_matrix(first_layer_size, no_of_features)
                        ),
                    ),
                ]
            )
        else:
            initialiser = INITIALISER_DICT[initialiser_key]

            self.encoder = tf.keras.Sequential(
                [
                    layers.Dense(
                        first_layer_size,
                        activation="relu",
                        use_bias=True,
                        kernel_initializer=initialiser,
                    ),
                    layers.Dense(
                        middle_layer_size,
                        activation="relu",
                        use_bias=True,
                        kernel_initializer=initialiser,
                    ),
                ]
            )

            self.decoder = tf.keras.Sequential(
                [
                    layers.Dense(
                        first_layer_size,
                        activation="relu",
                        use_bias=True,
                        kernel_initializer=initialiser,
                    ),
                    layers.Dense(
                        no_of_features,
                        activation="sigmoid",
                        use_bias=True,
                        kernel_initializer=initialiser,
                    ),
                ]
            )

    def decoded_latent_space(self, x: tf.Tensor) -> tf.Tensor:
        encoded = self.encoder.predict(x)
        decoded = self.decoder.predict(encoded)
        return decoded


def train_autoencoder(
    train_data_df: pd.DataFrame,
    test_data_df: pd.DataFrame,
    autoencoder_folder: str,
    no_of_epochs: int = 200,
    learning_rate: float = 0.0005,
    nodesize: int = 32,
    initialiser: str = "straddled",
    run_type: str = "all_layers",
    batch_size: int = FULL_BATCH_SIZE,
):
    """
    run_type options:
    - 'shuffled_feature_order' : shuffles feature order
    - any other string: use to mark experiment
    """

    try:
        train_data = train_data_df.to_numpy()
        test_data = test_data_df.to_numpy()
    except AttributeError as e:
        print(f"Data is already in np form: {e}")
        train_data = train_data_df
        test_data = test_data_df

    train_data = tf.cast(train_data, tf.float32)
    test_data_original = test_data.copy()
    test_data = tf.cast(test_data, tf.float32)

    no_of_features = train_data.shape[1]
    run_name = f"{run_type}_{nodesize}node_{initialiser}_{learning_rate}lr_{no_of_epochs}epochs"

    autoencoder = AnomalyDetector(
        FIRST_LAYER_SIZE, no_of_features, nodesize, initialiser
    )

    optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.0)
    autoencoder.compile(optimizer=optimizer, loss=root_mean_squared_error)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir="experiments/logs", write_grads=True, histogram_freq=1
    )

    LambdaCallback(
        on_epoch_end=lambda batch, logs: tf.summary.histogram(
            "Weights", autoencoder.get_weights()
        )
    )
    LambdaCallback(on_epoch_end=lambda batch, logs: print(autoencoder.get_weights()))

    history = autoencoder.fit(
        train_data,
        train_data,
        epochs=no_of_epochs,
        validation_data=(test_data, test_data),
        batch_size=batch_size,
        callbacks=[tensorboard_callback],
    )

    plt.figure()
    fig, ax = plt.subplots(1, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    min_value = min(history.history["val_loss"])
    min_index = history.history["val_loss"].index(min_value)
    plt.title(
        f"test loss: {round(min_value,3)}\ntrain loss: "
        f"{round(history.history['loss'][min_index],3)}\nat epoch {min_index}"
    )
    plt.ylim(0, 0.3)
    plt.legend()
    plt.gcf().subplots_adjust(top=0.85)
    fig.savefig(f"{autoencoder_folder}training_curves_{run_name}.png")
    plt.close()

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(f"{autoencoder_folder}training_curves_{run_name}.csv")

    autoencoder.save_weights(f"{autoencoder_folder}model_{run_name}")

    decoded_output = autoencoder.decoded_latent_space(test_data_original)
    SHOW_MNIST_OUTPUT = False
    if SHOW_MNIST_OUTPUT:
        n = 10  # How many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # Display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(test_data_original[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_output[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    return history


def create_outputs_for_runs(
    list_of_runs: list,
    experiment_name: str,
    experiment_path: str,
):
    runs_dict = {}
    for filename in list_of_runs:
        runs_dict[filename] = pd.read_csv(experiment_path + filename)

    plt.figure()
    fig, ax = plt.subplots(1, 1)
    colors = ["b", "g", "r", "c", "m", "y", "k", "orange"]
    for count, filename in enumerate(list_of_runs):
        plt.plot(
            runs_dict[filename]["loss"],
            c=colors[count],
            label="_".join(filename.split("_")[2:-1]),
        )
    for count, filename in enumerate(list_of_runs):
        plt.plot(runs_dict[filename]["val_loss"], "--", c=colors[count])
    plt.ylim(0, 0.08)
    plt.legend(fontsize=8)
    plt.title(experiment_name)
    plt.gcf().subplots_adjust(top=0.85)
    fig.savefig(f"{experiment_path}_experiment_{experiment_name}.png")
    plt.close()


def run_experiments(
    train: pd.DataFrame,
    test: pd.DataFrame,
    run_type: str,
    experiment_path: str,
    num_epochs: int,
    lr: float,
    middle_node_size: int,
    batch_size: int = FULL_BATCH_SIZE,
) -> list:
    train_data_df, test_data_df = train, test

    file = pathlib.Path(experiment_path)
    if not file.exists():
        raise FileExistsError("The experiment folder does not exist.")

    run_histories = []
    for key in INITIALISER_DICT:
        history = train_autoencoder(
            train_data_df,
            test_data_df,
            experiment_path,
            no_of_epochs=num_epochs,
            learning_rate=lr,
            nodesize=middle_node_size,
            initialiser=key,
            run_type=run_type,
            batch_size=batch_size,
        )
        run_histories.append((key, history))
    return run_histories


def process_experiments(name: str = "", experiment_path: str = ""):
    list_of_runs = []
    try:
        for key in INITIALISER_DICT:
            list_of_runs.append(
                f"training_curves_all_layers_32node_{key}_2000epochs.csv",
            )
        create_outputs_for_runs(
            list_of_runs, f"[{name}]different_initialisers_0p0001lr", experiment_path
        )
    except:
        print("No 2000 epochs")

    try:
        list_of_runs = []
        for key in INITIALISER_DICT:
            list_of_runs.append(f"training_curves_fasterLR_32node_{key}_1000epochs.csv")
        create_outputs_for_runs(
            list_of_runs, f"[{name}]different_initialisers_0p0002lr", experiment_path
        )
    except:
        print("No 1000 epochs")

    try:
        list_of_runs = []
        for key in INITIALISER_DICT:
            list_of_runs.append(
                f"training_curves_no_batching_32node_{key}_0.0005lr_200epochs.csv"
            )
        create_outputs_for_runs(
            list_of_runs, f"[{name}]different_initialisers_0p0005lr", experiment_path
        )
    except:
        print("No 200 epochs")

    try:
        list_of_runs = []
        for key in INITIALISER_DICT:
            list_of_runs.append(
                f"training_curves_no_batching_32node_{key}_0.001lr_50epochs.csv"
            )
        create_outputs_for_runs(
            list_of_runs, f"[{name}]different_initialisers_0p001lr", experiment_path
        )
    except:
        print("No 50 epochs")
