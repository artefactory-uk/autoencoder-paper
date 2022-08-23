from paths import CI_EXPERIMENT_PATH
import confidence_intervals as confidence_intervals
import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd

rc('font',**{'family':'serif','serif':['Founders Grotesk']})
plt.rcParams.update({'font.size': 29})

class MakePlots():
    def __init__(self, experiment_name, file_name):
        self.file_name = file_name
        self.experiment_name = experiment_name
        self.save_path =  CI_EXPERIMENT_PATH + self.experiment_name + '/'
        self.data_file_path = self.save_path + self.file_name + '_all_data.pkl'
        self.CI = confidence_intervals.ConfidenceIntervals()
        with open( self.data_file_path, 'rb') as pkl:
            self.all_data = pickle.load(pkl)

        self.num_epochs = self.all_data['num_epochs']
        self.epochs_list = self.all_data['epochs_list']
        self.train_losses = self.all_data['train_losses']
        self.val_losses = self.all_data['val_losses']
        self.num_experiments = self.all_data['num_experiments']
        self.color_map = ['blue', 'red','green','pink','orange','black','grey','yellow']


    def make_losses_dict(self, losses):
        epochs = dict()
        for i in range(self.num_epochs):
            epochs[i] = []
            for j in range(self.num_experiments):
                epochs[i].append(losses[i + (self.num_epochs * j)])
        return epochs

    def get_points_to_plot(self, key, val_losses):
        epochs_key = self.make_losses_dict(val_losses[key][0])
        epochs_straddled = self.make_losses_dict(val_losses['straddled'][0])
        key_lower, key_means, key_upper = self.CI.get_intervals(epochs_key)
        straddled_lower, straddled_means, straddled_upper = self.CI.get_intervals(epochs_straddled)
        return key_means, straddled_means

    def plot_epochs(self, axs, losses, name):
        first_epochs, last_epochs = [], []
        for cnt, key in enumerate(losses.keys()):
            all_losses = losses[key][0]
            epochs = self.make_losses_dict(all_losses)

            all_lower,all_means, all_upper = self.CI.get_intervals(epochs)

            # for plotting
            first_epochs.append(epochs[0])
            last_epochs.append(epochs[self.num_epochs - 1])


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

    def plot_all(self, epsilon, alpha):
        df = pd.DataFrame()
        converged_epochs, converged_losses = [], []
        fig, axs = plt.subplots(1, 1, figsize=(35, 20))

        for cnt, key in enumerate(self.val_losses.keys()):

            key_means_train, straddled_means_train = self.get_points_to_plot(key, self.train_losses)
            key_means_val, _ = self.get_points_to_plot(key, self.val_losses)
            converged_points = self.convergence_criteria(key_means_val,epsilon, alpha)
            axs.plot(self.epochs_list, key_means_val, label=key + ' validation',  linewidth=3,
                     color = self.color_map[cnt])

            if converged_points != False:
                converged_epoch, converged_loss = converged_points[0], converged_points[1]
                print(converged_epoch, converged_loss)
                axs.scatter(converged_epoch, converged_loss, s=500, marker = 'd',
                            color = self.color_map[cnt])
                converged_epochs.append(converged_epoch)
                converged_losses.append(converged_loss)
            else:
                converged_epochs.append(np.nan)
                converged_losses.append(np.nan)

            # axs.plot(self.epochs_list, key_means_train, label=key + ' train')
            # axs[0].fill_between(epochs_list, all_lower, all_upper, alpha=.2)
        axs.legend(loc="upper right")
        axs.set_title(f"{self.file_name}\n [$ \epsilon = {epsilon}$, $\\alpha = {alpha}$]")

        axs.set_xlabel('Epoch')
        axs.set_ylabel('Loss')
        df['Initialiser'] = list(self.val_losses.keys())
        df['Converged Epochs'] = converged_epochs
        df['Converged Loss'] = converged_losses
        fig.savefig(self.save_path+f'{self.experiment_name}_all_initialisers.png')
        df = df.sort_values(by=['Converged Loss','Converged Epochs'])
        return df.to_latex(index=False, label = f'{self.experiment_name} all initialisers')

    def pair_plot_epochs(self):
        for cnt, key in enumerate(self.val_losses.keys()):
            fig, axs = plt.subplots(1, 1, figsize=(30, 30))

            key_means_train, straddled_means_train = self.get_points_to_plot(key, self.train_losses)
            key_means_val, straddled_means_val = self.get_points_to_plot(key, self.val_losses)

            axs.plot(self.epochs_list, key_means_train, label=key + ' train',  color = 'blue')
            axs.plot(self.epochs_list, key_means_val, label=key + ' validation', color = 'blue', linestyle='dashed')

            axs.plot(self.epochs_list, straddled_means_train, label='straddled train',  color = 'red')
            axs.plot(self.epochs_list, straddled_means_val, label='straddled validation', color = 'red', linestyle='dashed')

            # axs[0].fill_between(epochs_list, all_lower, all_upper, alpha=.2)
            axs.legend(loc="upper right")
            axs.set_title(f"Straddled + {key}\n{self.file_name}")

            axs.set_xlabel('Epoch')
            axs.set_ylabel('Loss')
            fig.savefig(CI_EXPERIMENT_PATH+self.save_path+f'straddled_{key}_'+self.file_name + '.png')

    def convergence_criteria(self, loss_curve, epsilon, num_epochs):
        for cnt, epoch in enumerate(loss_curve):
            num_converged_epochs = 0
            for next_epoch in loss_curve[cnt:]:
                if next_epoch <= (epoch + epsilon) and next_epoch >= (epoch - epsilon):
                    num_converged_epochs += 1
                    if num_converged_epochs >= num_epochs:
                        print('Converged')
                        return cnt + 1, epoch
                else:
                    break

        return False

if __name__ == '__main__':
    plot_synthetic, plot_swarm, plot_mnist = False, False, True

    if plot_synthetic:
        plots = MakePlots('Synthetic Experiment TEST',
                          'Synthetic Experiment TEST\n[Straddled type = asymmetric | Num. Epochs = 1000 | Learning rate = 0.1 | Num. runs = 10]')

        converged_df = plots.plot_all(epsilon = 0.001, alpha = 100)
        print('Synthetic')
        print(converged_df)


    if plot_swarm:
        plots = MakePlots('Swarm Experiment TEST',
                          'Swarm Experiment TEST\n[Straddled type = asymmetric | Num. Epochs = 1500 | Learning rate = 0.1 | Num. runs = 1]')

        converged_df = plots.plot_all(epsilon = 0.005, alpha = 500)
        print('Swarm Behaviour')
        print(converged_df)

    if plot_mnist:
        plots = MakePlots('MNIST Experiment TEST',
                          'MNIST Experiment TEST\n[Straddled type = asymmetric | Num. Epochs = 1000 | Learning rate = 0.1 | Num. runs = 1]')

        converged_df = plots.plot_all(epsilon = 0.005, alpha = 250)
        print('MNIST')

        print(converged_df)
