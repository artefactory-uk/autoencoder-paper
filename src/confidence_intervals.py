import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from paths import CI_EXPERIMENT_PATH

class ConfidenceIntervals():
    def __init__(self, all_histories, num_experiments):
        self.all_histories = all_histories
        self.num_experiments = num_experiments
        self.num_epochs = len(self.all_histories[0][0][1].history['val_loss'])
        print(f'Number of epochs: {self.num_epochs}')

    def __calculate_CI(self,curve_points, CI = 0.95):
        '''
        Calculates CI for one epoch in the learning curve.
        '''
        a = 1.0 * np.array(curve_points)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + CI) / 2., n - 1)
        print(f'{m}, {m - h}, {m + h}')
        return m, m - h, m + h

    def calculate_CI_learning_curves(self):
        '''
        Calculates CIs for the entire learning curve.
        '''
        val_losses = dict()
        for history in self.all_histories:
            for experiment in history:
                curr_history = experiment[1].history
                init_type = experiment[0]
                val_loss = curr_history['val_loss']
                if init_type not in val_losses.keys():
                    val_losses[init_type] = []

                for cnt, loss in enumerate(val_loss):
                    if val_losses[init_type] != []:
                        val_losses[init_type][0].append(loss)
                    else:
                        val_losses[init_type].append([loss])

        for key in val_losses.keys():
            all_losses = val_losses[key][0]
            epochs = dict()
            print(f'CI for {key}')
            for i in range(self.num_epochs):
                epochs[i] = []
                for j in range(self.num_experiments):
                    print(f'Epoch {j}')
                    epochs[i].append(all_losses[i + (self.num_epochs * j)])
            print(epochs[0])
            all_means, all_upper, all_lower = [], [], []

            epochs_list = [str(i+1) for i in list(range(self.num_epochs))]

            for epoch in epochs.keys():
                mean, upper, lower = self.__calculate_CI(epochs[epoch])
                all_means.append(mean),all_upper.append(upper),all_lower.append(lower)
            plt.plot(epochs_list, all_means, label = key)
            plt.fill_between(epochs_list, all_lower, all_upper,  alpha=.2)
            plt.title(f"Confidence Interval for {key}")
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loc="upper right")
            plt.savefig(CI_EXPERIMENT_PATH+'CI_'+key+'.png')
            plt.clf()