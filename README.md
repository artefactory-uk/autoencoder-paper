# Autoencoder weight initialisation paper

Training neural networks is a complex task with many interdependent hyper-parameters that each have to undergo a non-intuitive tuning
process. Weight initialisation is generally overlooked in this process but
can have a drastic impact on performance. We propose a nonstochashtic
weight initialisation scheme that is an extension of the standard matrix
identity and show that when benchmarked with a simple autoencoder
on various datasets that our initialiser outperforms the current state of
the art as measured by both convergence time and loss reached.

## Setup

### Requirements

When you first close this repo run the following commands to properly initialise your local development environment

1. Create you local virtual environment (change python version as required):
```shell
python3.9 -m venv _venv
```

2. Activate and run only from the virtual environment:
```shell
source _venv/bin/activate
```

3. Install the requirements:
```shell
pip install -r requirements.txt
```

### Data

- `Synthetic`: created locally
- `MNIST`: availible via Keras
- `Swarm Behaviour`: needs to be downloaded from [this](https://www.kaggle.com/datasets/deepcontractor/swarm-behaviour-classification) link and moved into `autoencoder-paper/resources/swarmBehaviour`.

## Running the experiments
Each experiment has a config file named `X_experiments_config.json`, where X is one of {swarm, mnist, synthetic} with the following format:

```json
{
     "X Experiment":
  [
    {
      "num_tests": 10,
      "num_epochs": 1500,
      "learning_rate": 0.1
    }
  ]

}
```
> Note: `num_tests` is the number of times the experiment is run with different random seeds.

To run the experiment for dataset `X` you run the following:
```bash
python experiment_scripts/X_experiments.py
```

### Plots and tables

The main output of the experiments are the following figure and table generated in [make_plots.py](src/make_plots.py)

#### Main figure

<img src="https://user-images.githubusercontent.com/37986581/186881692-68e8d6f6-d72a-4251-9567-b1b7e2f6e278.png" width="500">
