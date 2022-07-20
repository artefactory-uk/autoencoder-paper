import src.confidence_intervals as confidence_intervals
import src.autoencoder_mnist_train as mnist

'''
This script runs experiments on the MNIST dataset 
'''

MNIST_CONFIG = \
    {
        'num_tests': 2, # How many tests for CI testing
        'sample_size': 0.05, # How much of the data to use (1 - sample_size = validation_size)
        'num_epochs': 2,
        'learning_rate': 0.001
    }

if __name__ == "__main__":
    name = f"TEST MNIST data [old (asymmetric)straddled but now using RMSE, using {MNIST_CONFIG['sample_size'] *100}% of training data]"

    seeds = list(range(MNIST_CONFIG['num_tests']))
    all_histories = []
    for seed in seeds:
        # all_histories.append(glove.run_glove(seed = seed))
        all_histories.append(mnist.run_mnist(seed = seed, sample_size=MNIST_CONFIG['sample_size'],
                                             num_epochs=MNIST_CONFIG['num_epochs'], lr=MNIST_CONFIG['learning_rate']))
        # all_histories.append(synthetic.run_synthetic(seed = seed))

    CIs = confidence_intervals.ConfidenceIntervals(all_histories, MNIST_CONFIG['num_tests'], name = name)
    CIs.calculate_CI_learning_curves()
