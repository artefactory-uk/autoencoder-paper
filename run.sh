#!/bin/bash
# Bash script to generate all experiment results
export PYTHONPATH=./
echo "Installing requirements"
pip install -r requirements.txt
echo "Running SYNTHETIC experiment"
python src/experiment_scripts/synthetic_experiments.py
echo "Running MNIST experiment"
python src/experiment_scripts/mnist_experiments.py
echo "Running SWARM experiment"
python src/experiment_scripts/swarm_experiments.py
echo "Generating Plots"
python src/make_plots.py
