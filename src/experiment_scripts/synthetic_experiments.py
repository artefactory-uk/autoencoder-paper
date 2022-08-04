import src.confidence_intervals as confidence_intervals
import src.autoencoder_synthetic_train as synthetic
import json
import time
import os
from src.paths import CI_EXPERIMENT_PATH

'''
This script runs experiments on the synthetic dataset 
'''
dir_path = os.path.abspath(os.path.dirname(__file__))
SYNTHETIC_CONFIG_FILENAME = dir_path+'/synthetic_experiments_config.json'
with open(SYNTHETIC_CONFIG_FILENAME) as config_file:
    all_experiments = json.load(config_file)
    all_experiments_names = all_experiments.keys()

if __name__ == "__main__":
    all_start_time = time.perf_counter()

    for experiment_name in all_experiments_names:
        print(f'Running: {experiment_name}')
        folder_name = CI_EXPERIMENT_PATH +experiment_name
        try:
            os.mkdir(folder_name, 0o777)
        except:
            print(f'Folder {folder_name} already exists.')

        for config in all_experiments[experiment_name]:
            name = f"{experiment_name}\n" \
                   f"[Straddled type = asymmetric | " \
                   f"Num. Epochs = {config['num_epochs']} | Learning rate = {config['learning_rate']} | " \
                   f"Num. runs = {config['num_tests']}]" \

            seeds = list(range(config['num_tests']))
            all_histories = []

            for seed in seeds:
                start_time = time.perf_counter()

                all_histories.append(synthetic.run_synthetic(seed = seed, num_epochs=config['num_epochs'],
                                                             lr=config['learning_rate']))
                end_time = time.perf_counter()
                print(f'{"-"*20}\n{round((end_time-start_time)/60,3)} minutes for 1 run of:\n{name}\n{"-"*20}\n')

            CIs = confidence_intervals.ConfidenceIntervals(all_histories, config['num_tests'], name = name, save_path = experiment_name +'/' )
            CIs.calculate_CI_learning_curves()

    all_end_time = time.perf_counter()
    print(f'{"-" * 20}\n{round((all_end_time - all_start_time) / 60, 3)} minutes for all synthetic experiments')