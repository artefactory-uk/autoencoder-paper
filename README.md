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

<img src="https://user-images.githubusercontent.com/67425173/232788168-2af1cd9d-562f-4aa5-aad0-08e0fec9e3d3.jpg"  width=50% height=50%>

The Straddled Matrix is defined simply by the function `straddled_matrix()` in [autoencoder_model.py](src/autoencoder_model.py).


## 🚀 What can this help with?

If you are looking for an alternative weight initialisation to improve the performance of your autoencoder, then this is for you! Designed for mostly linear datasets, but also works well with high amounts of non-linearity.

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
- `MNIST`: available via Keras
- `Swarm Behaviour`: can be downloaded from [here](https://www.kaggle.com/datasets/deepcontractor/swarm-behaviour-classification), and then copied to `autoencoder-paper/resources/swarmBehaviour`.

## 🚀 Running the experiments
To run all the experiments run the following command:
```bash
bash run.sh
```

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
Note: `num_tests` is the number of times the experiment is run with different random seeds. The MNIST and Swarm Behaviour experiments take a significant time. Consider reducing `num_tests` if you only want to test the experiments.


## 📊 Plots and tables

All the figures and tables that appear in the paper are generated inside the folder `experiments/plots`.
These are generated after running `make_plots.py` via the  `run.sh` bash script.


#### E.g.: figure for synthetic data experiment!

<img src="https://user-images.githubusercontent.com/67425173/232788521-6431ac29-27bc-4b16-8301-3a1aa884fcf1.jpeg"  width=50% height=50%>
