import src.autoencoder_glove_train as glove
import src.confidence_intervals as confidence_intervals

'''
This script runs experiments on the GLoVE dataset 
'''

GLOVE_CONFIG = \
    {
        'num_tests': 2, # How many tests for CI testing
        'sample_size': 0.05, # How much of the data to use (1 - sample_size = validation_size)
        'num_epochs': 130,
        'learning_rate': 0.005
    }

if __name__ == "__main__":
    name = f"GloVe data"
    seeds = list(range(GLOVE_CONFIG['num_tests']))
    all_histories = []
    for seed in seeds:
        all_histories.append(glove.run_glove(seed = seed, num_epochs=GLOVE_CONFIG['num_epochs'], lr=GLOVE_CONFIG['learning_rate']))

    CIs = confidence_intervals.ConfidenceIntervals(all_histories, GLOVE_CONFIG['num_tests'], name = name)
    CIs.calculate_CI_learning_curves()