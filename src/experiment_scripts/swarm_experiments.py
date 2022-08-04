import src.confidence_intervals as confidence_intervals
import src.autoencoder_swarm_behaviour_train as swarm
import time
import os
import helpers

'''
This script runs experiments on the Swarm Behaviour dataset 
'''
dir_path = os.path.abspath(os.path.dirname(__file__))
CONFIG_FILENAME = dir_path+'/swarm_experiments_config.json'

all_experiments, all_experiments_names = helpers.read_config_file(CONFIG_FILENAME)

if __name__ == "__main__":
    all_start_time = time.perf_counter()

    for experiment_name in all_experiments_names:
        print(f'Running: {experiment_name}')
        helpers.make_experiment_dir(experiment_name)

        for config in all_experiments[experiment_name]:
            name = helpers.construct_name(config, experiment_name)

            seeds = list(range(config['num_tests']))
            all_histories = []

            for seed in seeds:
                start_time = time.perf_counter()

                all_histories.append(swarm.run_swarm(seed = seed, num_epochs=config['num_epochs'],
                                                             lr=config['learning_rate']))
                end_time = time.perf_counter()
                helpers.print_one_run_time(start_time, end_time, name)

            CIs = confidence_intervals.ConfidenceIntervals(all_histories, config['num_tests'], name = name, save_path = experiment_name +'/' )
            CIs.calculate_CI_learning_curves()

    all_end_time = time.perf_counter()
    helpers.print_total_time(all_start_time, all_end_time, "swarm")