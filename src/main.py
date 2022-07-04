import autoencoder_glove_train as glove
import confidence_intervals
import autoencoder_synthetic_train as synthetic

'''
End to end pipeline for running the autoencoder on both the synthetic and glove data
'''
if __name__ == "__main__":
    NUM_TESTS = 4
    seeds = list(range(NUM_TESTS))
    all_histories = []
    for seed in seeds:
        all_histories.append(glove.run_glove(seed = seed))


    CIs = confidence_intervals.ConfidenceIntervals(all_histories, NUM_TESTS)
    CIs.calculate_CI_learning_curves()

    # synthetic.run_synthetic()