# Pokemon Hackathon

## 0. Overview

This project was created for a pokemon hackathon using kedro ([link](https://kedro.readthedocs.io)).
Goal of the challenge is
1. to predict the outcome of a battle between two pokemons of a given level and
2. using the predictive model to - given a budget limit - build a team of 6 pokemons from a list of 
available pokemons that performs optimal against 6 fixed opponent pokemons.


## 1. Setup

To prepare the environment for this project, you need to take the following
steps:
1. create a virtual environment ([link](https://docs.python.org/3/library/venv.html)) and activate it.
2. clone this repository and navigate to its root directory
3. run the following command in the command line to install all requirements

```
kedro install
```
## 2. High-level Approach

Our approach generally consists of the following steps:

First, we preprocessed the available training data. Second, we trained multiple deep neural networks (DNN)
for predicting the outcome of the battle between two pokemons. As our final prediction, we used an
ensemble of the neural networks.
Using the trained neural network, we predicted the performance of each available pokemon against the
opponent pokemons. Using these predictions, we tested different approaches (integer programming,
genetic algorithms) for finding an optimal team of pokemons given a budget constraint.

We implemented our approach as `kedro` pipelines which, we will outline in the next section.

## 3. Available Pipelines

To solve the task of the challenge, we implemented the following modular pipelines:

### 3.1. `preprocessing_pipeline`

The preprocessing pipeline prepares all input data for training of the predictive model as well
as for prediction of the performance of the pokemons for building a team.

### 3.2. `dnn_pipeline`

Once the preprocessing is done, the `dnn_pipeline` trains a deep neural network on the battle
training data given parameters in `./conf/base/parameters.yml`.

### 3.3. `eval_dnn_pipeline`

With all dnn_models trained, the `eval_dnn_pipeline` evaluates the performance of all models as
a standard ensemble and a weighted average ensemble (achieved by linear regression) on a holdout set.

### 3.4. `prediction_pipeline`

This pipelines uses the best performing model, which is the weighted ensemble, to predict
the performance of each available pokemon against the opponent pokemons.

### 3.5. `lp_optimization_pipeline` / `ga_optimization_pipeline`

These two pipelines are responsible for the final submission. Each of the two uses a different
approach (linear programming or genetic algorithm) to identify the best performing team of
6 pokemons using the predicitons from the previous given the budget constraint.

## 4. Running Pipelines

Each pipeline can be run individually using the following command in the command line:

```
kedro run --pipeline {name_of_pipeline}
```

For sake of convenience, we have included the pretrained DNN model in `./data/99_non_catalogued/`.
Therefore, to reproduce our results, `dnn_pipeline` and `eval_dnn_pipeline` do not have to be executed.

Hence, consecutively running `preprocessing_pipeline`, `prediction_pipeline` and `lp_optimization_pipeline` or
`ga_optimization_pipeline` will produce a submission file in `./data/08_reporting/`.
In our experiments, linear optimization produced the best results.

As an alternative to running the pipelines individually, we have implementation the
consecutive execution of `preprocessing_pipeline`, `prediction_pipeline` and `lp_optimization_pipeline`
as the default. Therefore, running the following command in the command line will also produce
our submission file:

```
kedro run
```

## 5. Authors

* Philipp Nikolaus (ETH Zurich) 
* Daniel Paysan (ETH Zurich)