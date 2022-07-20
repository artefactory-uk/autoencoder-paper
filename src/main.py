import autoencoder_glove_train as glove
import confidence_intervals
import autoencoder_synthetic_train as synthetic
import autoencoder_mnist_train as mnist


'''
End to end pipeline for running the autoencoder on both the synthetic and glove data
'''
if __name__ == "__main__":
    NUM_TESTS = 3
    SAMPLE_SIZE = 0.05
    name = f"MNIST data [old (asymmetric)straddled but now using RMSE, using {SAMPLE_SIZE *100}% of training data]"

    seeds = list(range(NUM_TESTS))
    all_histories = []
    for seed in seeds:
        # all_histories.append(glove.run_glove(seed = seed))
        all_histories.append(mnist.run_mnist(seed = seed, sample_size=SAMPLE_SIZE))
        # all_histories.append(synthetic.run_synthetic(seed = seed))


    CIs = confidence_intervals.ConfidenceIntervals(all_histories, NUM_TESTS, name = name)
    CIs.calculate_CI_learning_curves()

    # synthetic.run_synthetic()