# Straddled Matrix: Using linear initialisation to improve speed of convergence and fully-trained error in Autoencoders

## [[Paper]](about:blank)

## 🤔 What is this?
Training neural networks is a complex task with many interdependent hyper-parameters that each have to undergo a non-intuitive tuning
process. Weight initialisation is generally overlooked in this process but
can have a drastic impact on performance. We propose the <strong>Straddled Matrix</strong>, a nonstochashtic
weight initialisation scheme that is an extension of the standard matrix
identity and show that when benchmarked with a simple autoencoder
on various datasets that our initialiser outperforms the current state of
the art as measured by both convergence time and loss reached.

## 💊 The Straddled Matrix

The Straddled Matrix is a modification of the standard identity matrix where the zero padding is replaced by diagonally filling all rows. This ensures that every feature in the dataset receives equal weight, regardless of the network architecture.

The Straddled Matrix is defined simply by the function `straddled_matrix()` in [autoencoder_model.py](src/autoencoder_model.py).


## 🚀 What can this help with?

If you are looking for an alternative weight initilisation to improve the performance of your autoencoder, then this is for you! Designed for mostly linear datasets, but also works well with high amounts of non-linearity.

## 🌲 Environment Setup

1. Create a local virtual environment (change python version as required):
```shell
python3.9 -m venv _venv
```

2. Activate and run from the virtual environment:
```shell
source _venv/bin/activate
```

3. Install the requirements:
```shell
pip install -r requirements.txt
```

## 📚 Datasets used

- `Synthetic`: created locally
- `MNIST`: availible via Keras
- `Swarm Behaviour`: can be downloaded from [here](https://www.kaggle.com/datasets/deepcontractor/swarm-behaviour-classification), and then copied to `autoencoder-paper/resources/swarmBehaviour`.

## 🚀 Running the experiments
Each experiment presented in the paper has a config file named `X_experiments_config.json`, where X is one of {swarm, mnist, synthetic} with the following format:

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
Note: `num_tests` is the number of times the experiment is run with different random seeds.

To run the experiment for dataset `X` run the following:
```bash
python experiment_scripts/X_experiments.py
```

## 📊 Plots and tables

Once the experiments have been run, generate the figures and tables that appear in the paper by running:
```shell
python make_plots.py
```

#### E.g.: figure for synthetic data experiment
