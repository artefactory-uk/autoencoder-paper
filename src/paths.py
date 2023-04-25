from pathlib import Path

ROOT = Path.cwd()
SOURCE_PATH = f"{ROOT}/src"

EXPERIMENT_FOLDER = "experiments"
LOGS_PATH = f"{SOURCE_PATH}/{EXPERIMENT_FOLDER}/logs/"
PLOTS_EXPERIMENT_PATH = f"{SOURCE_PATH}/{EXPERIMENT_FOLDER}/plots/"

SWARM_DATA_PATH = f"{SOURCE_PATH}/resources/swarmBehaviour/Swarm_Behaviour.csv"

SYNTHETIC_EXPERIMENT_PATH = f"{SOURCE_PATH}/{EXPERIMENT_FOLDER}/synthetic/"
MNIST_EXPERIMENT_PATH = f"{SOURCE_PATH}/{EXPERIMENT_FOLDER}/mnist/"
SWARM_EXPERIMENT_PATH = f"{SOURCE_PATH}/{EXPERIMENT_FOLDER}/swarmBehaviour/"

SYNTHETIC_CONFIG_PATH = (
    f"{SOURCE_PATH}/experiment_scripts/synthetic_experiments_config.json"
)
MNIST_CONFIG_PATH = f"{SOURCE_PATH}/experiment_scripts/mnist_experiments_config.json"
SWARM_CONFIG_PATH = f"{SOURCE_PATH}/experiment_scripts/swarm_experiments_config.json"
