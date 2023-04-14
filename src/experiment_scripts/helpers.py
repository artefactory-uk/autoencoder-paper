import json
from pathlib import Path

from src.paths import PLOTS_EXPERIMENT_PATH


def read_config_file(file_path: str) -> (dict, list):
    with open(file_path) as config_file:
        all_experiments = json.load(config_file)
        all_experiments_names = all_experiments.keys()
    return all_experiments, all_experiments_names


def construct_name(config: dict, experiment_name: str) -> str:
    name = (
        f"{experiment_name}"
        f"[Straddled type = asymmetric | "
        f"Num. Epochs = {config['num_epochs']} | Learning rate = {config['learning_rate']} | "
        f"Num. runs = {config['num_tests']}]"
    )
    return name


def make_experiment_dir(experiment_name: str):
    path = PLOTS_EXPERIMENT_PATH + experiment_name
    path_to_check = Path(path)
    if not path_to_check.is_dir():
        path_to_check.mkdir(parents=True, exist_ok=False)
        print(f"Folder created for {path_to_check}")


def make_missing_dir(experiment_name: str):
    path = experiment_name
    path_to_check = Path(path)
    if not path_to_check.is_dir():
        path_to_check.mkdir(parents=True, exist_ok=False)
        print(f"Folder created for {path_to_check}")


def print_one_run_time(start_time: float, end_time: float, name: str):
    print(
        f'{"-" * 20}\n{round((end_time - start_time) / 60, 3)} minutes for 1 run of:\n{name}\n{"-" * 20}\n'
    )


def print_total_time(all_start_time: float, all_end_time: float, dataset_name: str):
    print(
        f'{"-" * 20}\n{round((all_end_time - all_start_time) / 60, 3)} minutes for all {dataset_name} experiments'
    )
