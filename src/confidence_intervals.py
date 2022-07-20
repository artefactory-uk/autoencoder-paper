import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from paths import CI_EXPERIMENT_PATH

class ConfidenceIntervals():
    def __init__(self, all_histories, num_experiments,name):
        self.all_histories = all_histories
        self.name = name
        self.num_experiments = num_experiments
        self.num_epochs = len(self.all_histories[0][0][1].history['val_loss'])
        self.CI = 0.95
        print(f'Number of epochs: {self.num_epochs}')

    def __calculate_CI(self,curve_points, CI = 0.95):
        '''
        Calculates CI for one epoch in the learning curve.
        '''
        points = np.array(curve_points)
        num_points = len(points)
        mean, standard_error = np.mean(points), scipy.stats.sem(points)
        lower_bound, upper_bound = scipy.stats.t.interval(CI, num_points -1, loc = mean, scale = standard_error)
        return mean, lower_bound, upper_bound

    def __restruct_history(self, key):
        history_dict = dict()
        for history in self.all_histories:
            for experiment in history:
                curr_history = experiment[1].history
                init_type = experiment[0]
                value = curr_history[key]

                if init_type not in history_dict.keys():
                    history_dict[init_type] = []

                for cnt, loss in enumerate(value):
                    if history_dict[init_type] != []:
                        history_dict[init_type][0].append(loss)
                    else:
                        history_dict[init_type].append([loss])

        return history_dict

    def plot_epochs(self, axs, val_losses, name):
        first_epochs, last_epochs = [], []
        for cnt, key in enumerate(val_losses.keys()):
            all_losses = val_losses[key][0]
            epochs = dict()
            for i in range(self.num_epochs):
                epochs[i] = []
                for j in range(self.num_experiments):
                    print(f'Epoch {j}')
                    epochs[i].append(all_losses[i + (self.num_epochs * j)])

            all_means, all_upper, all_lower = [], [], []

            # for plotting
            epochs_list = [int(i + 1) for i in list(range(self.num_epochs))]
            first_epochs.append(epochs[0])
            last_epochs.append(epochs[self.num_epochs - 1])

            for epoch in epochs.keys():
                mean, upper, lower = self.__calculate_CI(epochs[epoch], CI=self.CI)
                all_means.append(mean), all_upper.append(upper), all_lower.append(lower)

            axs[cnt].plot(epochs_list, all_means, label=name)
            axs[cnt].fill_between(epochs_list, all_lower, all_upper, alpha=.2)
            axs[cnt].set_title(f"{key}")
            # axs[cnt].set_ylim((0.04, 0.15))
            axs[cnt].set_xlabel('Epoch')
            axs[cnt].set_ylabel('Loss')
            axs[cnt].legend(loc="upper right")

        # Reset axis
        print(first_epochs)
        y_max, y_min = np.max(first_epochs), np.min(last_epochs)
        for cnt, _ in enumerate(val_losses.keys()):
            axs[cnt].set_ylim((y_min - (0.01*y_min), y_max+(0.01*y_max)))

    def calculate_CI_learning_curves(self):
        '''
        Calculates CIs for the entire learning curve.
        '''
        val_losses = self.__restruct_history('val_loss')
        losses = self.__restruct_history('loss')


        fig, axs = plt.subplots(1, len(val_losses.keys()), figsize=(38.5, 7.5))


        self.plot_epochs(axs, losses, "train loss")
        self.plot_epochs(axs, val_losses,"validation loss")

        fig.suptitle(f"{self.name} {self.CI*100}% Confidence interval [{self.num_epochs} epochs, {self.num_experiments} runs]")
        fig.savefig(CI_EXPERIMENT_PATH + self.name +'_CI_' + 'test' + '.png')

