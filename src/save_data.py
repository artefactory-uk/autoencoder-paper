import pandas as pd
import pickle

from src.paths import PLOTS_EXPERIMENT_PATH


class SaveData:
    def __init__(
        self, all_histories: list, num_experiments: int, name: str, save_path: str = ""
    ):
        self.all_histories = all_histories
        self.name = name
        self.num_experiments = num_experiments
        self.num_epochs = len(self.all_histories[0][0][1].history["val_loss"])
        self.epochs_list = [int(i + 1) for i in list(range(self.num_epochs))]

        self.save_path = save_path
        self.CI = 0.95

    def __restruct_history(self, key: str) -> dict:
        history_dict = {}
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

    def __history_to_csv(self, train_losses: dict, val_losses: dict):
        history_df = pd.DataFrame()
        history_df["epoch"] = self.epochs_list

        for key in train_losses.keys():
            for i in range(self.num_experiments):
                history_df[key + " train - run " + str(i + 1)] = train_losses[key][0][
                    i * self.num_epochs : ((i + 1) * self.num_epochs)
                ]
                history_df[key + " validation - run " + str(i + 1)] = val_losses[key][
                    0
                ][i * self.num_epochs : ((i + 1) * self.num_epochs)]

        history_df.to_csv(
            PLOTS_EXPERIMENT_PATH + self.save_path + self.name + ".csv", index=False
        )

    def __save_pkl(self, all_data_dict: dict):
        save_path = PLOTS_EXPERIMENT_PATH + self.save_path + self.name + "_all_data.pkl"

        with open(save_path, "wb") as pkl:
            pickle.dump(all_data_dict, pkl)

    def save_all_data(self):
        """
        Calculates CIs for the entire learning curve.
        """
        val_losses = self.__restruct_history("val_loss")
        losses = self.__restruct_history("loss")

        print("Saving experiment history...")
        self.__history_to_csv(losses, val_losses)
        all_data = dict(
            {
                "train_losses": losses,
                "val_losses": val_losses,
                "num_epochs": self.num_epochs,
                "epochs_list": self.epochs_list,
                "num_experiments": self.num_experiments,
            }
        )
        self.__save_pkl(all_data)
