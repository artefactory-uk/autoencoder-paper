import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt
from matplotlib import rc

from src.paths import CI_EXPERIMENT_PATH

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Founders Grotesk']})
plt.rcParams.update({'font.size': 29})

class ConfidenceIntervals():
    def __init__(self, all_histories, num_experiments,name, save_path = ""):
        self.all_histories = all_histories
        self.name = name
        self.num_experiments = num_experiments
        self.num_epochs = len(self.all_histories[0][0][1].history['val_loss'])
        self.epochs_list = [int(i + 1) for i in list(range(self.num_epochs))]

        self.save_path = save_path
        self.CI = 0.95

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

    def plot_epochs(self, axs, losses, name):
        first_epochs, last_epochs = [], []
        for cnt, key in enumerate(losses.keys()):
            all_losses = losses[key][0]
            epochs = self.make_losses_dict(all_losses)

            all_means, all_upper, all_lower = [], [], []

            # for plotting
            first_epochs.append(epochs[0])
            last_epochs.append(epochs[self.num_epochs - 1])

            for epoch in epochs.keys():
                mean, upper, lower = self.__calculate_CI(epochs[epoch], CI=self.CI)
                all_means.append(mean), all_upper.append(upper), all_lower.append(lower)

            axs[cnt].plot(self.epochs_list, all_means, label=name)
            axs[cnt].fill_between(self.epochs_list, all_lower, all_upper, alpha=.2)
            axs[cnt].set_title(f"{key}")
            # Plot lowest point
            lowest = min(zip(np.round(all_means,3),self.epochs_list))
            lowest_loss,lowest_epoch = lowest[0], lowest[1]

            if name == 'train loss':
                color = 'blue'
            else:
                color = 'orange'

            axs[cnt].scatter(lowest_epoch, lowest_loss,color = color, s= 90)
            axs[cnt].axhline(y=lowest_loss, linestyle='--',color = color, label = f"Lowest point ({name}) = {round(lowest_loss,3)} at epoch {lowest_epoch}")

            # axs[cnt].set_ylim((0.04, 0.15))            axs[cnt].plot(lowest_epoch, lowest_loss)
            axs[cnt].set_xlabel('Epoch')
            axs[cnt].set_ylabel('Loss')
            # axs[cnt].legend(loc="upper right")

        # Reset axis
        y_max, y_min = np.max(first_epochs), np.min(last_epochs)
        for cnt, _ in enumerate(losses.keys()):
            axs[cnt].set_ylim((y_min - (0.04*y_min), y_max+(0.01*y_max)))

    def pair_plot_epochs(self, train_losses, val_losses):
        for cnt, key in enumerate(val_losses.keys()):
            fig, axs = plt.subplots(1, 1, figsize=(30, 30))

            key_means_train, straddled_means_train = self.get_points_to_plot(key, train_losses)
            key_means_val, straddled_means_val = self.get_points_to_plot(key, val_losses)

            axs.plot(self.epochs_list, key_means_train, label=key + ' train',  color = 'blue')
            axs.plot(self.epochs_list, key_means_val, label=key + ' validation', color = 'blue', linestyle='dashed')

            axs.plot(self.epochs_list, straddled_means_train, label='straddled train',  color = 'red')
            axs.plot(self.epochs_list, straddled_means_val, label='straddled validation', color = 'red', linestyle='dashed')

            # axs[0].fill_between(epochs_list, all_lower, all_upper, alpha=.2)
            axs.legend(loc="upper right")
            axs.set_title(f"Straddled + {key}\n{self.name}")

            axs.set_xlabel('Epoch')
            axs.set_ylabel('Loss')
            fig.savefig(CI_EXPERIMENT_PATH+self.save_path+f'straddled_{key}_'+self.name + '.png')

    def get_points_to_plot(self, key, val_losses):
        epochs_key = self.make_losses_dict(val_losses[key][0])
        epochs_straddled = self.make_losses_dict(val_losses['straddled'][0])
        key_lower, key_means, key_upper = self.get_intervals(epochs_key)
        straddled_lower, straddled_means, straddled_upper = self.get_intervals(epochs_straddled)
        return key_means, straddled_means

    def get_intervals(self, epochs_key):

        all_means, all_upper, all_lower = [], [], []
        for epoch in epochs_key.keys():
            mean, upper, lower = self.__calculate_CI(epochs_key[epoch], CI=self.CI)
            all_means.append(mean), all_upper.append(upper), all_lower.append(lower)
        return all_lower, all_means, all_upper



    def __history_to_csv(self, train_losses, val_losses):
        history_df = pd.DataFrame()
        history_df['epoch'] = self.epochs_list

        for key in train_losses.keys():
            for i in range(self.num_experiments):
                history_df[key + ' train - run '+ str(i + 1)] = train_losses[key][0][i * self.num_epochs :((i+1) *self.num_epochs)]
                history_df[key + ' validation - run '+str(i + 1)] = val_losses[key][0][i * self.num_epochs :((i+1) *self.num_epochs)]

        history_df.to_csv(CI_EXPERIMENT_PATH+self.save_path + self.name+'.csv', index = False)

    def make_losses_dict(self, losses):
        epochs = dict()
        for i in range(self.num_epochs):
            epochs[i] = []
            for j in range(self.num_experiments):
                epochs[i].append(losses[i + (self.num_epochs * j)])
        return epochs

    def calculate_CI_learning_curves(self):
        '''
        Calculates CIs for the entire learning curve.
        '''
        val_losses = self.__restruct_history('val_loss')
        losses = self.__restruct_history('loss')
        try:
            self.__history_to_csv(losses, val_losses)
            print('csv saved successfully')
        except Exception as e:
            print('Saving csv failed')
            print(e)

        fig, axs = plt.subplots(1, len(val_losses.keys()), figsize=(100, 15.5))

        self.plot_epochs(axs, losses, "train loss")
        self.plot_epochs(axs, val_losses,"validation loss")
        self.pair_plot_epochs(train_losses = losses, val_losses = val_losses)

        fig.suptitle(f"{self.name}\n{self.CI*100}% Confidence interval\n\n")
        fig.savefig(CI_EXPERIMENT_PATH + self.save_path + self.name +'_CI_' + 'test' + '.png')


