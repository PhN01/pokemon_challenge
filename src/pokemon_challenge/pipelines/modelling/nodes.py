import logging
from typing import Any, Dict
import time
import random
import os
import copy
import json
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

import ortools
from ortools.linear_solver import pywraplp

import torch
import torch.utils.data as data_utils
from torch import nn
from torch.optim import AdamW

from tqdm import tqdm

class torch_regression_model():

    def __init__(
            self,
            dnn_arch,
            output_dir,
            num_epochs=None,
            early_stopping=None,
            input_dim=None,
            weight_decay=None,
            learning_rate=None
    ):

        self.dnn_arch = dnn_arch
        self.optimizer = None
        self.num_epochs = num_epochs
        self.loss_function = nn.MSELoss()
        self.early_stopping = early_stopping
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

        self.input_dim = input_dim
        self.output_dir = output_dir

        self.device = self.get_device()
        self.model = None

        self.fitting_history_dict = None

    def get_device(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return device

    def init_model(self):

        assert not any([param is None for param in [self.num_epochs, self.early_stopping, self.input_dim]])

        log = logging.getLogger(__name__)

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        torch.manual_seed(1234)

        model = nn.Sequential()
        model.add_module('dense_0', nn.Linear(self.input_dim, self.dnn_arch[0]))
        for i in range(1, len(self.dnn_arch)):
            model.add_module('dense_{}'.format(i), nn.Linear(self.dnn_arch[i - 1], self.dnn_arch[i]))
            model.add_module('norm_{}'.format(i), nn.BatchNorm1d(self.dnn_arch[i]))
            model.add_module('relu_{}'.format(i), nn.ReLU())
        model.add_module('out', nn.Linear(self.dnn_arch[-1], 1))

        model.apply(init_weights)

        params_to_update = model.parameters()
        self.optimizer = AdamW(params_to_update, lr=self.learning_rate, weight_decay=self.weight_decay)

        self.model = model

        log.info(f"Torch model with architecture {self.dnn_arch} initialized.")


    def train(self, data_loaders_dict):

        log = logging.getLogger(__name__)

        assert not self.model is None, "Model not initialized. Call init_model() or load_model() before running train()."

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", verbose=True
        )

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        device = self.get_device()

        since = time.time()

        self.model = self.model.to(device)

        val_loss_history = []
        train_loss_history = []
        self.fitting_history_dict = {"train": train_loss_history, "val": val_loss_history}

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = np.infty
        early_stopping_counter = 0

        for epoch in range(self.num_epochs):
            log.info(f"Epoch {epoch}/{self.num_epochs-1}")
            log.info("-" * 70)

            if early_stopping_counter > self.early_stopping and early_stopping_counter > 0:
                log.info(f"Stopped training because of no improvement of the validation score for "
                         f"{self.early_stopping} epochs")
                break

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_error = 0.0

                for index, data in enumerate(data_loaders_dict[phase]):
                    inputs = data[0].type(torch.FloatTensor).to(device)
                    labels = data[1].type(torch.FloatTensor).to(device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Enable training
                    with torch.set_grad_enabled(
                            phase == "train"
                    ) and torch.autograd.set_detect_anomaly(False):

                        # Forward pass and calculate loss
                        outputs = self.model(inputs)
                        batch_size = labels.size(0)
                        loss = self.loss_function(
                            outputs.view(batch_size, -1), labels.view(batch_size, -1)
                        )

                    # Backpropagation of the loss during the training phase
                    if phase == "train":
                        with torch.autograd.set_detect_anomaly(False):
                            loss.backward()
                            self.optimizer.step()

                    # Compute epoch statistics.
                    running_loss += loss.item()
                    running_error += np.sqrt(running_loss)

                epoch_loss = running_loss / len(data_loaders_dict[phase])
                epoch_roloss = np.sqrt(running_loss / len(data_loaders_dict[phase]))

                log.info(f"{phase} {loss.__class__.__name__} loss: {epoch_loss:.6f} root of loss: {epoch_roloss:.6f}")

                # Deep copy the model if it has the best validation loss.
                # Thereby the best validation loss and not the potentially requested square root of it is used to determine
                # the superiority of a model. Due to the concavity of the square root function this has no influence
                # on the overall process.
                if phase == "val":
                    scheduler.step(epoch_loss)
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_model = copy.deepcopy(self.model)
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        torch.save(best_model, f"{self.output_dir}/best_model.pth")
                        torch.save(best_model_wts, f"{self.output_dir}/best_model_weights.pth")
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1

                self.fitting_history_dict[phase].append(epoch_loss)

        time_elapsed = time.time() - since
        log.info(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

        log.info(f"Best val loss : {best_loss:4f}")

        # with open(f"{self.output_dir}/train_hist.json", "w") as f:
        #     json.dump(self.fitting_history_dict, f)

        # Load best model weights
        self.model.load_state_dict(best_model_wts)

        # Get test loss
        if "test" in data_loaders_dict.keys():
            running_loss = 0.0
            running_error = 0.0
            for index, data in enumerate(data_loaders_dict["test"]):
                inputs = data[0].type(torch.FloatTensor).to(device)
                labels = data[1].type(torch.FloatTensor).to(device)

                with torch.set_grad_enabled(False):
                    # Forward pass
                    outputs = self.model(inputs)
                    outputs = outputs.view(-1)
                    labels = labels.view(-1)
                    loss = self.loss_function(outputs, labels)

                # Compute statistics.
                running_loss += loss.item()
                running_error += np.sqrt(running_loss)

            epoch_loss = running_loss / len(data_loaders_dict["test"])
            epoch_roloss = np.sqrt(running_loss / len(data_loaders_dict["test"]))

            log.info(f"test {loss.__class__.__name__} loss: {epoch_loss:.6f} root of loss: {epoch_roloss:.6f}")
            log.info("-" * 70)
            log.info("-" * 70)

    def load_model(self):

        assert os.path.isdir(self.output_dir)
        assert "best_model.pth" in os.listdir(self.output_dir)

        if torch.cuda.is_available():
            self.model = torch.load(f"{self.output_dir}/best_model.pth")
        else:
            self.model = torch.load(f"{self.output_dir}/best_model.pth", map_location=torch.device('cpu'))



def get_predictions_and_true_labels(
        data_loader, fitted_model
):
    r""" Function to get the true labels and the predictions by a fitted model for a provided dataset.

    Parameters
    ----------
    data_loader : :py:class:`~torch.utils.data.Dataloader`
        The `Dataloader` that operates on the dataset for which the representations are supposed to be derived.

    fitted_model : :py:class:`~torch.nn.Module`
        The fitted model instance.

    device : :py:class:`~torch.device.Device`
        The device to cun the computations on.

    Returns
    -------
    (predictions, true_labels) : tuple(numpy.ndarray, numpy.ndarray)
        [1] The predictions obtained from the fitted model for the provided dataset.
        [2] The respective true labels of the same samples.

    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    fitted_model.to(device)
    fitted_model.eval()

    predictions = []
    true_labels = []

    for index, data in enumerate(data_loader):
        inputs = data[0].type(torch.FloatTensor).to(device)
        labels = data[1].type(torch.FloatTensor).to(device)
        preds = fitted_model(inputs)
        preds = preds.view(-1)
        predictions.append(np.array(preds.detach().cpu().numpy()))

        labels = labels.view(-1)
        true_labels.append(np.array(labels.detach().cpu().numpy()))

    predictions = np.array([item for sublist in predictions for item in sublist])
    predictions = predictions.reshape(-1)
    true_labels = np.array([item for sublist in true_labels for item in sublist])
    true_labels.reshape(-1)

    return predictions, true_labels


def get_predictions(
        data_loader, fitted_model
):
    r""" Function to get the true labels and the predictions by a fitted model for a provided dataset.

    Parameters
    ----------
    data_loader : :py:class:`~torch.utils.data.Dataloader`
        The `Dataloader` that operates on the dataset for which the representations are supposed to be derived.

    fitted_model : :py:class:`~torch.nn.Module`
        The fitted model instance.

    device : :py:class:`~torch.device.Device`
        The device to cun the computations on.

    Returns
    -------
    (predictions, true_labels) : tuple(numpy.ndarray, numpy.ndarray)
        [1] The predictions obtained from the fitted model for the provided dataset.
        [2] The respective true labels of the same samples.

    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    fitted_model.to(device)
    fitted_model.eval()

    predictions = []

    for index, data in enumerate(data_loader):
        inputs = data[0].type(torch.FloatTensor).to(device)
        preds = fitted_model(inputs)
        preds = preds.view(-1)
        predictions.append(np.array(preds.detach().cpu().numpy()))

    predictions = np.array([item for sublist in predictions for item in sublist])
    predictions = predictions.reshape(-1)

    return predictions


def eval_ensemble_for_dataset(model_dict, dataloader):
    dataset_summary = {}
    all_preds = []

    for model_path, model in tqdm(model_dict.items(), desc='Models'):
        model_name = model_path.split("/")[-1]
        dataset_summary[model_name] = {}

        preds, true_labels = get_predictions_and_true_labels(dataloader, model.model)
        dataset_summary[model_name]['preds'] = preds.tolist()
        all_preds.append(preds)

        dataset_summary[model_name]['mse'] = float(mean_squared_error(true_labels, preds))

    return dataset_summary, all_preds, true_labels


def predict_ensemble_for_dataset(model_dict, dataloader):
    all_preds = []

    for model in model_dict.values():
        preds = get_predictions(dataloader, model.model)
        all_preds.append(preds)

    return all_preds


###################################################
# Nodes
###################################################

def fit_dnn_model(
        X_train: pd.DataFrame, y_train: pd.DataFrame,
        X_val: pd.DataFrame, y_val: pd.DataFrame,
        X_test: pd.DataFrame, y_test: pd.DataFrame,
        dnn_params: dict, dnn_arch: dict
):
    log = logging.getLogger(__name__)

    np.random.seed(1234)
    random.seed(1234)
    torch.manual_seed(1234)

    if torch.cuda.is_available():
        train_tensors = data_utils.TensorDataset(
            torch.cuda.FloatTensor(np.array(X_train)),
            torch.cuda.FloatTensor(np.array(y_train)))

        val_tensors = data_utils.TensorDataset(
            torch.cuda.FloatTensor(np.array(X_val)),
            torch.cuda.FloatTensor(np.array(y_val)))

        test_tensors = data_utils.TensorDataset(
            torch.cuda.FloatTensor(np.array(X_test)),
            torch.cuda.FloatTensor(np.array(y_test)))
    else:
        train_tensors = data_utils.TensorDataset(
            torch.FloatTensor(np.array(X_train)),
            torch.FloatTensor(np.array(y_train)))

        val_tensors = data_utils.TensorDataset(
            torch.FloatTensor(np.array(X_val)),
            torch.FloatTensor(np.array(y_val)))

        test_tensors = data_utils.TensorDataset(
            torch.FloatTensor(np.array(X_test)),
            torch.FloatTensor(np.array(y_test)))


    train_loader = data_utils.DataLoader(train_tensors,
                                       batch_size = 1024, shuffle = True)

    val_loader = data_utils.DataLoader(val_tensors,
                                       batch_size = 1024, shuffle = False)

    test_loader = data_utils.DataLoader(test_tensors,
                                       batch_size = 1024, shuffle = False)

    data_loaders_dict = {'train':train_loader, 'val':val_loader, 'test':test_loader}


    torch_net = torch_regression_model(
        dnn_arch=dnn_arch['arch'],
        num_epochs=dnn_params['num_epochs'],
        early_stopping=dnn_params['early_stopping'],
        output_dir=dnn_arch['output_dir'],
        input_dim=X_train.shape[1]
    )
    torch_net.init_model()

    torch_net.train(data_loaders_dict)


def eval_dnn_models(
        X_train: pd.DataFrame, y_train: pd.DataFrame,
        X_val: pd.DataFrame, y_val: pd.DataFrame,
        X_test: pd.DataFrame, y_test: pd.DataFrame,
        dnn_arch_1: dict, dnn_arch_2: dict, dnn_arch_3: dict, dnn_arch_4: dict
):
    log = logging.getLogger(__name__)

    if torch.cuda.is_available():
        train_tensors = data_utils.TensorDataset(
            torch.cuda.FloatTensor(np.array(X_train)),
            torch.cuda.FloatTensor(np.array(y_train)))

        val_tensors = data_utils.TensorDataset(
            torch.cuda.FloatTensor(np.array(X_val)),
            torch.cuda.FloatTensor(np.array(y_val)))

        test_tensors = data_utils.TensorDataset(
            torch.cuda.FloatTensor(np.array(X_test)),
            torch.cuda.FloatTensor(np.array(y_test)))
    else:
        train_tensors = data_utils.TensorDataset(
            torch.FloatTensor(np.array(X_train)),
            torch.FloatTensor(np.array(y_train)))

        val_tensors = data_utils.TensorDataset(
            torch.FloatTensor(np.array(X_val)),
            torch.FloatTensor(np.array(y_val)))

        test_tensors = data_utils.TensorDataset(
            torch.FloatTensor(np.array(X_test)),
            torch.FloatTensor(np.array(y_test)))

    train_loader = data_utils.DataLoader(train_tensors,
                                              batch_size=1024, shuffle=False)

    val_loader = data_utils.DataLoader(val_tensors,
                                       batch_size=1024, shuffle=False)

    test_loader = data_utils.DataLoader(test_tensors,
                                        batch_size=1024, shuffle=False)

    data_loaders_dict = {
        'train':train_loader,
        'val':val_loader,
        'test':test_loader
    }

    model_dict = {}
    for dnn in [dnn_arch_1,dnn_arch_2,dnn_arch_3,dnn_arch_4]:
        model = torch_regression_model(dnn['arch'], output_dir=dnn['output_dir'])
        model.load_model()
        model_dict[dnn['output_dir']] = model

    model_summaries_dict = {}

    log.info("Starting model evaluation.")
    for dataset, dataloader in tqdm(data_loaders_dict.items(), desc='Datasets'):

        dataset_summary, all_preds, true_labels = eval_ensemble_for_dataset(model_dict, dataloader)

        model_summaries_dict[dataset] = dataset_summary

        all_preds = np.array(all_preds)
        avg_preds = np.mean(all_preds, axis=0)
        model_summaries_dict[dataset]['ensemble'] = {}
        model_summaries_dict[dataset]['ensemble']['preds'] = avg_preds.tolist()
        model_summaries_dict[dataset]['ensemble']['mse'] = float(mean_squared_error(true_labels, avg_preds))

        # train ensemble linear regression for weighted averaging
        if dataset == 'train':
            global lm
            log.info("Training linear regression model on individual model predictions.")
            lm = LinearRegression(n_jobs=-1)
            lm.fit(all_preds.transpose(), true_labels)

        weighted_avg_preds = lm.predict(all_preds.transpose())
        model_summaries_dict[dataset]['weighted_ensemble'] = {}
        model_summaries_dict[dataset]['weighted_ensemble']['preds'] = weighted_avg_preds.tolist()
        model_summaries_dict[dataset]['weighted_ensemble']['mse'] = float(mean_squared_error(true_labels, weighted_avg_preds))

    log.info("Storing evaluation summary.")
    with open("./data/99_non_catalogued/dnn_evaluation.json", 'w') as f:
        json.dump(model_summaries_dict, f)

    log.info("Storing ensemble regression model.")
    with open("./data/99_non_catalogued/ensemble_regression.pkl", 'wb') as f:
        pickle.dump(lm, f)

    for dataset in data_loaders_dict.keys():
        log.info(f"Performance evaluation for training data.")
        for model_path in model_dict.keys():
            model_name = model_path.split("/")[-1]
            log.info(f"Model {model_name}.  MSE: {model_summaries_dict[dataset][model_name]['mse']}")
        log.info(f"Ensemble.  MSE: {model_summaries_dict[dataset]['ensemble']['mse']}")
        log.info(f"Weighted Ensemble.  MSE: {model_summaries_dict[dataset]['weighted_ensemble']['mse']}")


def predict_available_pokemon_performance(
        available_pokemon: pd.DataFrame, battles: pd.DataFrame, #train_battles: pd.DataFrame,
        dnn_arch_1: dict, dnn_arch_2: dict, dnn_arch_3: dict, dnn_arch_4: dict
):
    log = logging.getLogger(__name__)

    log.info("Preparing data and loading models.")
    if torch.cuda.is_available():
        data_tensor = data_utils.TensorDataset(torch.cuda.FloatTensor(np.array(battles)))

    else:
        data_tensor = data_utils.TensorDataset(torch.FloatTensor(np.array(battles)))

    dataloader = data_utils.DataLoader(data_tensor,
                                              batch_size=1024, shuffle=False)

    model_dict = {}
    for dnn in [dnn_arch_1,dnn_arch_2,dnn_arch_3,dnn_arch_4]:
        model = torch_regression_model(dnn['arch'], output_dir=dnn['output_dir'])
        model.load_model()
        model_dict[dnn['output_dir']] = model

    with open('./data/99_non_catalogued/ensemble_regression.pkl', 'rb') as f:
        lm = pickle.load(f)

    log.info("Creating predictions.")
    all_preds = predict_ensemble_for_dataset(model_dict, dataloader)
    all_preds = np.array(all_preds)

    log.info("Creating weighted average over predictions using ensemble regression model.")
    weighted_ensemble_preds = lm.predict(all_preds.transpose())

    enemy_pokemons = ['caterpie', 'golem', 'krabby', 'mewtwo', 'raichu', 'venusaur']

    for i, pokemon in enumerate(enemy_pokemons):
        available_pokemon[f'res_{pokemon}'.lower()] = weighted_ensemble_preds[i::6]
        available_pokemon[f'hppr_{pokemon}'.lower()] = np.minimum(
            np.maximum(
                available_pokemon[f'res_{pokemon}'.lower()], 0
            ) / available_pokemon['HP_1'], 1
        )

    return available_pokemon, pd.DataFrame(all_preds)

