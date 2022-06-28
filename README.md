# autoencoder-paper

## Initialising your local repo
When you first close this repo run the following commands to properly initialise your local development environment

1. Create you local virtual environment (change python version as required):
```shell
python3.9 -m venv _venv
```

2. Always activate and work only from the virtual environment:
```shell
source _venv/bin/activate
```

3. Install `pre-commit` hooks:
```shell
pre-commit install
```

## Default requirements
The `requirements.txt` file contains default requirements which are often relevant to data repository work.

The `test-requirements.txt` file contains default requirements which are often relevant for the testing of data
repository code.

Both of these requirements files are minimum recommendations but may be adjusted as necessary per project.

## DVC
When setting up DVC you must ensure that the `.dvc` files are located in the same folder as the file/folder they are
representing. Placing the `.dvc` folder in a different path to its target may cause issues during some updates.

## Set up your PyCharm IDE
1. Ensure your Python interpreter is set to your local `_venv` virtual environment.
You may need to restart PyCharm if you have updated you Python interpreter.
2. Ensure when you open your PyCharm terminal tab that the shell running inside the virtual environment.
You should see `(_venv)` at the start or end of the shell prompt.
