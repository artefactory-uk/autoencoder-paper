import pandas as pd
import numpy as np
from src.autoencoder_model import run_experiments, process_experiments, set_seeds

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from src.paths import SWARM_DATA_PATH, SWARM_EXPERIMENT_PATH


def load_swarm_data():
    swarm_df = pd.read_csv(SWARM_DATA_PATH)
    print(swarm_df.dtypes)
    cols_to_drop = ['Swarm_Behaviour']
    swarm_df = swarm_df.drop(columns = cols_to_drop)
    print(swarm_df.info())
    print(swarm_df['xVel1'])
    scaler = MinMaxScaler()
    swarm_df = scaler.fit_transform(swarm_df)


    return swarm_df

def run_swarm(seed, num_epochs, lr):
    set_seeds(seed)
    swarm_data = load_swarm_data()

    run_type = "no_batching"
    #
    data_size = len(swarm_data)
    train_frac = 0.8
    shuffled_swarm_data = shuffle(swarm_data)
    train_set, test_set = (
        shuffled_swarm_data[:int(data_size * train_frac)],
        shuffled_swarm_data[int(data_size * train_frac): ],
    )
    #
    run_histories = run_experiments(train_set, test_set, run_type=run_type, experiment_path=SWARM_EXPERIMENT_PATH,
                                    num_epochs=num_epochs, lr=lr)
    file_name = 'test_swarm'
    process_experiments(name=file_name, experiment_path=SWARM_EXPERIMENT_PATH)
    return run_histories

if __name__ == "__main__":
    run_swarm(1, 10 ,0.01)