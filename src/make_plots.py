import confidence_intervals as confidence_intervals
import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import numpy as np
import pandas as pd
import json
from scipy import stats

from src.paths import (
    PLOTS_EXPERIMENT_PATH,
    SYNTHETIC_CONFIG_PATH,
    MNIST_CONFIG_PATH,
    SWARM_CONFIG_PATH,
)
import experiment_scripts.helpers as helpers

rc("font", **{"family": "serif", "serif": ["Founders Grotesk"]})
plt.rcParams.update({"font.size": 29})


class MakePlots:
    def __init__(self, experiment_name: str, file_name: str):
        self.file_name = file_name
        self.experiment_name = experiment_name
        self.save_path = PLOTS_EXPERIMENT_PATH + self.experiment_name + "/"
        self.data_file_path = self.save_path + self.file_name + "_all_data.pkl"
        self.CI = confidence_intervals.ConfidenceIntervals()
        with open(self.data_file_path, "rb") as pkl:
            self.all_data = pickle.load(pkl)

        self.num_epochs = self.all_data["num_epochs"]
        self.epochs_list = self.all_data["epochs_list"]
        self.train_losses = self.all_data["train_losses"]
        self.val_losses = self.all_data["val_losses"]
        self.num_experiments = self.all_data["num_experiments"]
        self.color_map = [
            "blue",
            "red",
            "green",
            "pink",
            "orange",
            "black",
            "grey",
            "yellow",
        ]

    def make_losses_dict(self, losses: list) -> dict:
        epochs = {}
        for i in range(self.num_epochs):
            epochs[i] = []
            for j in range(self.num_experiments):
                epochs[i].append(losses[i + (self.num_epochs * j)])
        return epochs

    def get_points_to_plot(
        self, key: str, val_losses: dict
    ) -> (list, list, list, list):
        epochs_dict = self.make_losses_dict(val_losses[key][0])
        epochs_straddled = self.make_losses_dict(val_losses["straddled"][0])
        key_lower, key_means, key_upper = self.CI.get_intervals(epochs_dict)
        straddled_lower, straddled_means, straddled_upper = self.CI.get_intervals(
            epochs_straddled
        )
        return key_means, straddled_means, key_lower, key_upper

    def plot_all(self, epsilon: float, alpha: int) -> (pd.DataFrame, str):
        df = pd.DataFrame()
        converged_epochs, converged_losses = [], []
        plt.rcParams.update({"font.size": 45})
        fig, axs = plt.subplots(1, 1, figsize=(35, 20))

        for cnt, key in enumerate(self.val_losses.keys()):

            key_means_val, _, key_low_val, key_high_val = self.get_points_to_plot(
                key, self.val_losses
            )
            converged_points = self.convergence_criteria(key_means_val, epsilon, alpha)
            axs.plot(
                self.epochs_list,
                key_means_val,
                label=key,
                linewidth=4,
                color=self.color_map[cnt],
            )
            axs.fill_between(
                self.epochs_list,
                key_low_val,
                key_high_val,
                color=self.color_map[cnt],
                alpha=0.1,
            )

            if converged_points:
                converged_epoch, converged_loss = (
                    converged_points[0],
                    converged_points[1],
                )
                axs.scatter(
                    converged_epoch,
                    converged_loss,
                    s=600,
                    marker="d",
                    color=self.color_map[cnt],
                )
                converged_epochs.append(converged_epoch)
                converged_losses.append(converged_loss)
            else:
                converged_epochs.append(np.nan)
                converged_losses.append(np.nan)

        axs.scatter([], [], s=500, color="black", marker="d", label="convergence")
        axs.legend(loc="upper right")

        axs.axis(xmin=self.epochs_list[0], xmax=self.epochs_list[-1])
        axs.set_xlabel("Epoch", fontsize=60)
        axs.set_ylabel("Loss", fontsize=60)
        df["Initialiser"] = list(self.val_losses.keys())
        df["Converged Epochs"] = converged_epochs
        df["Converged Loss"] = converged_losses
        fig.savefig(self.save_path + f"{self.experiment_name}_all_initialisers.pdf")
        helpers.make_missing_dir(PLOTS_EXPERIMENT_PATH + f"main_figures/")
        fig.savefig(
            PLOTS_EXPERIMENT_PATH
            + f"main_figures/{self.experiment_name}_all_initialisers.pdf"
        )

        return (
            df,
            df.to_latex(index=False, label=f"{self.experiment_name} all initialisers"),
        )

    @staticmethod
    def convergence_criteria(
        loss_curve: list, epsilon: float, num_epochs: int
    ) -> (int, float):
        for cnt, epoch in enumerate(loss_curve):
            num_converged_epochs = 0
            for next_epoch in loss_curve[cnt:]:
                if (epoch + epsilon) >= next_epoch >= (epoch - epsilon):
                    num_converged_epochs += 1
                    if num_converged_epochs >= num_epochs:
                        return cnt + 1, epoch
                else:
                    break

    def plot_dist_final_loss(self, second_best: pd.DataFrame):
        """
        Plot the distribution of the final loss reached
        """
        second_best = second_best.iloc[1]["Initialiser"]
        epochs_glorot = self.make_losses_dict(self.val_losses[second_best][0])[
            self.num_epochs - 1
        ]
        epochs_straddled = self.make_losses_dict(self.val_losses["straddled"][0])[
            self.num_epochs - 1
        ]

        fig, axs = plt.subplots(1, 1, figsize=(30, 30))
        sns.histplot(epochs_glorot, label=second_best, ax=axs, kde=True, color="red")
        sns.histplot(
            epochs_straddled, label="straddled", ax=axs, kde=True, color="blue"
        )
        axs.legend(loc="upper right")
        p_value = np.round(stats.ttest_ind(epochs_straddled, epochs_glorot)[1], 3)
        print(
            f"Furthermore, performing a t-test on the distribution of the losses at the final epoch "
            f"between straddled and the next best initialiser ({second_best}) revealed p-value of  {p_value} "
        )
        fig.savefig(self.save_path + f"last_epoch_dist.png")


def print_experiment_title(name: str):
    delim = "-" * (len(name) * 2)
    space = " " * (len(name) // 2)
    print(f"{delim}\n{space}{name}\n{delim}\n")


def print_experiment_subtitle(name: str):
    delim = "=" * (len(name) // 2)
    print(f"\n{delim} {name} {delim}\n")


def display_experiment(title: str, dir_name: str, epsilon: float, alpha: int):
    plots = MakePlots(dir_name, title)
    converged_df = plots.plot_all(epsilon=epsilon, alpha=alpha)
    print_experiment_title(dir_name)
    print_experiment_subtitle(f"t-test ({dir_name})")
    plots.plot_dist_final_loss(converged_df[0])
    print_experiment_subtitle(f"convergence latex table ({dir_name})")
    print(converged_df[1])
    print_experiment_subtitle(f"latex caption for main figure ({dir_name})")
    print(f"{title}")


if __name__ == "__main__":
    plot_synthetic, plot_swarm, plot_mnist = True, True, True

    with open(SYNTHETIC_CONFIG_PATH) as config_file:
        all_experiments = json.load(config_file)
        experiment_details = all_experiments["Synthetic Experiment"][0]
        num_epochs_synthetic = experiment_details["num_epochs"]
        num_runs_synthetic = experiment_details["num_tests"]
    with open(MNIST_CONFIG_PATH) as config_file:
        all_experiments = json.load(config_file)
        experiment_details = all_experiments["MNIST Experiment"][0]
        num_epochs_mnist = experiment_details["num_epochs"]
        num_runs_mnist = experiment_details["num_tests"]
    with open(SWARM_CONFIG_PATH) as config_file:
        all_experiments = json.load(config_file)
        experiment_details = all_experiments["Swarm Experiment"][0]
        num_epochs_swarm = experiment_details["num_epochs"]
        num_runs_swarm = experiment_details["num_tests"]

    if plot_synthetic:
        display_experiment(
            title=f"Synthetic Experiment[Straddled type = asymmetric | Num. Epochs = {num_epochs_synthetic} | "
            f"Learning rate = 0.1 | Num. runs = {num_runs_synthetic}]",
            dir_name="Synthetic Experiment",
            epsilon=0.001,
            alpha=100,
        )

    if plot_mnist:
        display_experiment(
            title=f"MNIST Experiment[Straddled type = asymmetric | Num. Epochs = {num_epochs_mnist} | "
            f"Learning rate = 0.1 | Num. runs = {num_runs_mnist}]",
            dir_name="MNIST Experiment",
            epsilon=0.005,
            alpha=250,
        )

    if plot_swarm:
        display_experiment(
            title=f"Swarm Experiment[Straddled type = asymmetric | "
            f"Num. Epochs = {num_epochs_swarm} | Learning rate = 0.1 | Num. runs = {num_runs_swarm}]",
            dir_name="Swarm Experiment",
            epsilon=0.005,
            alpha=500,
        )
