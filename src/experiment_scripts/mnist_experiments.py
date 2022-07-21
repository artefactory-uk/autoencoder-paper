import src.confidence_intervals as confidence_intervals
import src.autoencoder_mnist_train as mnist
import json

'''
This script runs experiments on the MNIST dataset 
'''
MNIST_CONFIG_FILENAME = 'experiment_scripts/mnist_experiments_config.json'
with open(MNIST_CONFIG_FILENAME) as config_file:
    all_experiments = json.load(config_file)
    all_experiments_names = all_experiments.keys()

if __name__ == "__main__":

    for experiment_name in all_experiments_names:
        print(f'Running: {experiment_name}')
        for config in all_experiments[experiment_name]:
            name = f"{experiment_name}\n" \
                   f"[Straddled type = asymmetric | Sample size = {config['sample_size'] * 100}% | " \
                   f"Num. Epochs = {config['num_epochs']} | Learning rate = {config['learning_rate']} | " \
                   f"Num. runs = {config['num_tests']}]" \

            seeds = list(range(config['num_tests']))
            all_histories = []
            for seed in seeds:
                all_histories.append(mnist.run_mnist(seed = seed, sample_size=config['sample_size'],
                                                     num_epochs=config['num_epochs'], lr=config['learning_rate']))

            CIs = confidence_intervals.ConfidenceIntervals(all_histories, config['num_tests'], name = name)
            CIs.calculate_CI_learning_curves()
