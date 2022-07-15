import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from paths import SYNTHETIC_EXPERIMENT_PATH

from src.autoencoder_helper_functions import scale_dataset
from src.autoencoder_model import (
    set_seeds,
    run_experiments,
    INITIALISER_DICT,
    create_outputs_for_runs,
)

FIRST_DIM = 20
FINAL_DIM = 100
DATASET_SIZE = 5000


def transform_vector(input_vector):
    prng2 = np.random.RandomState(2019)
    unfolded_vector = np.repeat(input_vector, repeats=FINAL_DIM, axis=0)
    possible_funcs = [
        lambda x: 0,
        lambda x: 0,
        lambda x: 0,
        lambda x: 0,
        lambda x: 0,
        lambda x: x - 0.5,
        lambda x: x + 0.1,
        lambda x: x / 2,
        lambda x: x * -1,
        lambda x: x,
        lambda x: np.sin(x),
        lambda x: x**2,
        lambda x: np.exp(x),
    ]
    unfolded_func_matrix = prng2.choice(
        possible_funcs, FIRST_DIM * FINAL_DIM, replace=True
    )
    apply_vectorized = np.vectorize(lambda f, x: f(x), otypes=[object])
    transformed_values = apply_vectorized(unfolded_func_matrix, unfolded_vector)
    transformed_values_2d = np.reshape(transformed_values, (FIRST_DIM, FINAL_DIM))
    transformed_values_1d = transformed_values_2d.sum(axis=0)
    transformed_values_vector = transformed_values_1d.reshape(-1, 1)
    return transformed_values_vector


def create_synthetic_data():
    prng = np.random.RandomState(60)
    dataset = np.empty((FINAL_DIM, 0), int)
    for i in range(DATASET_SIZE):
        random_vector = prng.randn(
            FIRST_DIM,
        )
        dataset = np.append(dataset, transform_vector(random_vector), axis=1)
    dataset = np.swapaxes(dataset, 0, 1)
    return pd.DataFrame.from_records(dataset)


def prepare_synthetic_data(data):
    scaled_data = scale_dataset(
        data, "autoencoder_synthetic", SYNTHETIC_EXPERIMENT_PATH, rescale=True
    )
    train, test = train_test_split(scaled_data, test_size=0.2)
    return train, test


def run_experiments_synthetic():
    run_type = "synthetic_100_features"
    dataset = create_synthetic_data()
    train_data_df, test_data_df = prepare_synthetic_data(dataset)
    histories = run_experiments(train_data_df, test_data_df, run_type, SYNTHETIC_EXPERIMENT_PATH)

    return histories


def process_experiments_synthetic():
    list_of_runs = []
    for key in INITIALISER_DICT:
        list_of_runs.append(
            f"training_curves_synthetic_100_features_32node_{key}_0.001lr_50epochs.csv"
        )
    create_outputs_for_runs(
        list_of_runs, "different_initialisers_0p001lr_synthetic_100_features",
        SYNTHETIC_EXPERIMENT_PATH
    )

def run_synthetic(seed):
    set_seeds(seed)
    run_histories = run_experiments_synthetic()
    process_experiments_synthetic()
    return run_histories

if __name__ == "__main__":
    run_synthetic()
