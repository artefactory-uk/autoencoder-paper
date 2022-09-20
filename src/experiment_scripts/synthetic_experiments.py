import time
import os
import helpers

import src.save_data as save_data
import src.autoencoder_synthetic_train as synthetic
from src.paths import SYNTHETIC_EXPERIMENT_PATH

dir_path = os.path.abspath(os.path.dirname(__file__))
SYNTHETIC_CONFIG_FILENAME = dir_path + "/synthetic_experiments_config.json"

all_experiments, all_experiments_names = helpers.read_config_file(
    SYNTHETIC_CONFIG_FILENAME
)

if __name__ == "__main__":
    """
    This script runs experiments on the Synthetic dataset
    """
    all_start_time = time.perf_counter()

    for experiment_name in all_experiments_names:
        print(f"Running: {experiment_name}")
        helpers.make_experiment_dir(experiment_name)
        helpers.make_missing_dir(SYNTHETIC_EXPERIMENT_PATH)

        for config in all_experiments[experiment_name]:
            name = helpers.construct_name(config, experiment_name)

            seeds = list(range(config["num_tests"]))
            all_histories = []

            for run_number, seed in enumerate(seeds):
                start_time = time.perf_counter()

                all_histories.append(
                    synthetic.run_synthetic(
                        seed=seed,
                        num_epochs=config["num_epochs"],
                        lr=config["learning_rate"],
                        middle_node_size=config["middle_node_size"],
                    )
                )
                end_time = time.perf_counter()
                helpers.print_one_run_time(start_time, end_time, name)

                data_saver = save_data.SaveData(
                    all_histories,
                    run_number + 1,
                    name=name,
                    save_path=experiment_name + "/",
                )
                data_saver.save_all_data()

    all_end_time = time.perf_counter()
    helpers.print_total_time(all_start_time, all_end_time, "synthetic")
