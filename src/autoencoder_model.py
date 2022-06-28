import pandas as pd
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.backends.backend_pdf
from tensorflow.keras import layers, optimizers


tf.random.set_seed(0)
FIRST_LAYER_SIZE = 64
STRADDLED = True

INITIALISER_DICT = {
    "straddled": "straddled",
    "glorotnormal": tf.keras.initializers.GlorotNormal(),
    "glorotuniform": tf.keras.initializers.GlorotUniform(),
    "henormal": tf.keras.initializers.HeNormal(),
    "heuniform": tf.keras.initializers.HeUniform(),
    "identity": tf.keras.initializers.Identity(),
    "orthogonal": tf.keras.initializers.Orthogonal(),
    "random": tf.keras.initializers.RandomNormal(),
}

def straddled_matrix(shape1, shape2):
    small_matrix = np.identity(shape2)
    matrix = small_matrix
    for i in range(ceil(shape1 / shape2)):
        matrix = np.concatenate((matrix, small_matrix), axis=0)
    return matrix[:shape1, :] + abs(np.random.normal(0, 0.001, size=(shape1, shape2)))


class AnomalyDetector(tf.keras.Model):
    def __init__(
        self,
        first_layer_size,
        no_of_features,
        middle_layer_size,
        initialiser_key="straddled",
        run_type="all_layers",
    ):
        super().__init__()
        if initialiser_key == "straddled":
            self.encoder = tf.keras.Sequential(
                [
                    layers.Dense(
                        first_layer_size,
                        activation="relu",
                        kernel_initializer=tf.constant_initializer(
                            straddled_matrix(no_of_features, first_layer_size)
                        ),
                    ),
                    layers.Dense(
                        middle_layer_size,
                        activation="relu",
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
                        kernel_initializer=tf.constant_initializer(
                            straddled_matrix(middle_layer_size, first_layer_size)
                        ),
                    ),
                    layers.Dense(
                        no_of_features,
                        activation="sigmoid",
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
                        kernel_initializer=initialiser,
                    ),
                    layers.Dense(
                        middle_layer_size,
                        activation="relu",
                        kernel_initializer=initialiser,
                    ),
                ]
            )

            self.decoder = tf.keras.Sequential(
                [
                    layers.Dense(
                        first_layer_size,
                        activation="relu",
                        kernel_initializer=initialiser,
                    ),
                    layers.Dense(
                        no_of_features,
                        activation="sigmoid",
                        kernel_initializer=initialiser,
                    ),
                ]
            )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(
    train_data_df,
    test_data_df,
    autoencoder_folder,
    no_of_epochs=50,
    learning_rate=0.001,
    nodesize=32,
    initialiser="straddled",
    run_type="all_layers",
):
    """
    run_type options:
    - 'shuffled_feature_order' : shuffles feature order
    - any other string: use to mark experiment
    """

    train_data = train_data_df.to_numpy()
    test_data = test_data_df.to_numpy()

    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)

    no_of_features = train_data.shape[1]
    run_name = f"{run_type}_{nodesize}node_{initialiser}_{learning_rate}lr_{no_of_epochs}epochs"

    autoencoder = AnomalyDetector(
        FIRST_LAYER_SIZE, no_of_features, nodesize, initialiser, run_type
    )

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss="mae")

    history = autoencoder.fit(
        train_data,
        train_data,
        epochs=no_of_epochs,
        validation_data=(test_data, test_data),
        shuffle=True,
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

    def print_features_pdf(model, input_df, type, run_name):
        input_array = input_df.to_numpy()
        encoded_data = model.encoder(input_array).numpy()
        decoded_data = model.decoder(encoded_data).numpy()

        pdf = matplotlib.backends.backend_pdf.PdfPages(
            f"{autoencoder_folder}/autoencoder_{type}_{run_name}.pdf"
        )
        no_of_features_to_look_at = len(input_df.columns)

        mae = []
        for feature in range(no_of_features_to_look_at):
            mae.append(
                np.abs(
                    np.subtract(input_array[:, feature], decoded_data[:, feature])
                ).mean()
            )

        mae_df = pd.DataFrame(
            list(zip(input_df.columns, list(range(no_of_features_to_look_at)), mae)),
            columns=["feature_name", "feature_number", "MAE"],
        )

        for index, row in mae_df.sort_values("MAE", ascending=False).iterrows():
            feature_number = row["feature_number"]
            feature_name = row["feature_name"]
            MAE = row["MAE"]
            if True:
                fig, ax = plt.subplots(1, 1)
                ax = plt.scatter(
                    input_array[:, feature_number], decoded_data[:, feature_number]
                )
                plt.title(feature_name + f" ({type})\nMAE: {round(MAE,3)}")
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.xlabel("input")
                plt.ylabel("decoded output")
                pdf.savefig(fig)
        pdf.close()

    print_pdf_with_feature_reconstruction_plots = False
    if print_pdf_with_feature_reconstruction_plots:
        print_features_pdf(autoencoder, train_data_df, "train", run_name)
        print_features_pdf(autoencoder, test_data_df, "test", run_name)


def create_outputs_for_runs(list_of_runs, experiment_name, experiment_path):
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


def run_autoencoder(autoencoder_folder, data):
    no_of_features = data.shape[1]
    autoencoder = AnomalyDetector(FIRST_LAYER_SIZE, no_of_features, 32, STRADDLED)
    autoencoder.load_weights(f"{autoencoder_folder}model")

    return autoencoder.predict(data)


def run_experiments_batch1(train, test, run_type, experiment_path):
    train_data_df, test_data_df = train, test
    print(experiment_path)
    for key in INITIALISER_DICT:
        train_autoencoder(
            train_data_df,
            test_data_df,
            experiment_path,
            no_of_epochs=50,
            learning_rate=0.001,
            nodesize=32,
            initialiser=key,
            run_type=run_type,
        )


def process_experiments_batch1(name="", experiment_path=""):
    list_of_runs = []
    try:
        for key in INITIALISER_DICT:
            list_of_runs.append(
                f"training_curves_all_layers_32node_{key}_2000epochs.csv",
            )
        create_outputs_for_runs(
            list_of_runs, f"[{name}]different_initialisers_0p0001lr",
            experiment_path
        )
    except:
        print("No 2000 epochs")

    try:
        list_of_runs = []
        for key in INITIALISER_DICT:
            list_of_runs.append(f"training_curves_fasterLR_32node_{key}_1000epochs.csv")
        create_outputs_for_runs(
            list_of_runs, f"[{name}]different_initialisers_0p0002lr",
            experiment_path
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
            list_of_runs, f"[{name}]different_initialisers_0p0005lr",experiment_path
        )
    except:
        print("No 200 epochs")

    try:
        list_of_runs = []
        for key in INITIALISER_DICT:
            list_of_runs.append(
                f"training_curves_no_batching_32node_{key}_0.001lr_50epochs.csv"
            )
        create_outputs_for_runs(list_of_runs, f"[{name}]different_initialisers_0p001lr",experiment_path)
    except:
        print("No 50 epochs")