import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Sequence
from typing import Tuple
from typing import Callable
from abc import abstractmethod
import sklearn.metrics as sk_metrics
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import inspect
from typing import Callable, Mapping, Union, List
import random
import scipy
import argparse

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from src.architecture import *
from src.dataset import *
from src.utils import *
from src.nam_trainer import *

# Define NAM class with interaction terms
class NA2MBase:
    def __init__(
        self,
        units_multiplier: int = 2,
        num_basis_functions: int = 64,
        hidden_sizes: list = [64, 32],
        dropout: float = 0.1,
        feature_dropout: float = 0.05, 
        batch_size: int = 1024,
        num_workers: int = 0,
        num_epochs: int = 1000,
        log_dir: str = None,
        val_split: float = 0.15,
        device: str = 'cpu',
        lr: float = 0.02082,
        decay_rate: float = 0.995,
        output_reg: float = 0.2078,
        l2_reg: float = 0.0,
        save_model_frequency: int = 10,
        patience: int = 60,
        monitor_loss: bool = True,
        early_stop_mode: str = 'min',
        loss_func: Callable = None,
        metric: str = None,
        num_learners: int = 1,
        n_jobs: int = None,
        warm_start: bool = False,
        random_state: int = 42
    ) -> None:
        self.units_multiplier = units_multiplier
        self.num_basis_functions = num_basis_functions
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.log_dir = log_dir
        self.val_split = val_split
        self.device = device
        self.lr = lr
        self.decay_rate = decay_rate
        self.output_reg = output_reg
        self.l2_reg = l2_reg
        self.save_model_frequency = save_model_frequency
        self.patience = patience
        self.monitor_loss = monitor_loss
        self.early_stop_mode = early_stop_mode
        self.loss_func = loss_func
        self.metric = metric
        self.num_learners = num_learners
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.random_state = random_state

        self._best_checkpoint_suffix = 'best'
        self._fitted = False

    def _set_random_state(self):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        return
    
    def _initialize_models(self, X, y):
        self.num_tasks = y.shape[1] if len(y.shape) > 1 else 1
        self.num_inputs = X.shape[1]
        self.models = []
        for _ in range(self.num_learners):
            model = NA2M(num_inputs=self.num_inputs,
                num_units=[64] * self.num_inputs,
                dropout=self.dropout,
                feature_dropout=self.feature_dropout,
                hidden_sizes=self.hidden_sizes)
            self.models.append(model)

        return

    def _models_to_device(self, device):
        for model in self.models:
            model.to(device)

        return

    def fit(self, X, y, w=None):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            w = w.to_numpy()

        self._set_random_state()
        if not self.warm_start or not self._fitted:
            self._initialize_models(X, y)

        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, w=None) -> None:
        self._models_to_device(self.device)
        
        # self._preprocessor = MinMaxScaler(feature_range = (-1, 1))

        # dataset = NAMDataset(self._preprocessor.fit_transform(X), y, w)
        dataset = NAMDataset(X, y, w)

        self.criterion = make_penalized_loss_func(self.loss_func, 
            self.regression, self.output_reg, self.l2_reg)

        self.trainer = Trainer(
            models=self.models,
            dataset=dataset,
            metric=self.metric,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            num_epochs=self.num_epochs,
            log_dir=self.log_dir,
            val_split=self.val_split,
            test_split=None,
            device=self.device,
            lr=self.lr,
            decay_rate=self.decay_rate,
            save_model_frequency=self.save_model_frequency,
            patience=self.patience,
            monitor_loss=self.monitor_loss,
            early_stop_mode=self.early_stop_mode,
            criterion=self.criterion,
            regression=self.regression,
            num_learners=self.num_learners,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        
        self.trainer.train_ensemble()
        self.trainer.close()

        # Move models to cpu so predictions can be made on cpu data
        self._models_to_device('cpu')

        self._fitted = True
        return self

    def predict(self, X) -> ArrayLike:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        # X = self._preprocessor.transform(X)
        X = torch.tensor(X, requires_grad=False, dtype=torch.float)
        predictions = np.zeros((X.shape[0],))
        if self.num_tasks > 1:
            predictions = np.zeros((X.shape[0], self.num_tasks))

        for model in self.models:
            preds, _ = model.forward(X)
            predictions += preds.detach().cpu().numpy()

        # predictions = self._preprocessor.inverse_transform(predictions)
        return predictions / self.num_learners

    def plot(self, feature_index) -> None:
        num_samples = 1000
        X = np.zeros((num_samples, self.num_inputs))
        X[:, feature_index] = np.linspace(-1.0, 1.0, num_samples)
        
        feature_outputs = []
        for model in self.models:
            # (examples, tasks, features)
            _, fnns_out = model.forward(torch.tensor(X, dtype=torch.float32))
            if self.num_tasks == 1:
                fnns_out = fnns_out.unsqueeze(dim=1)
            # (examples, tasks)
            feature_outputs.append(fnns_out[:, :, feature_index].detach().cpu().numpy())

        # (learners, examples, tasks)
        feature_outputs = np.stack(feature_outputs, axis=0)
        # (examples, tasks)
        y = np.mean(feature_outputs, axis=0).squeeze()
        conf_int = np.std(feature_outputs, axis=0).squeeze()
        # TODO: Scale conf_int according to units of y

        # X = self._preprocessor.inverse_transform(X)
        
        return {'x': X[:, feature_index], 'y': y, 'conf_int': conf_int}

    def load_checkpoints(self, checkpoint_dir):
        self.models = []
        for i in range(self.num_learners):
            checkpointer = Checkpointer(os.path.join(checkpoint_dir, str(i)))
            model = checkpointer.load(self._best_checkpoint_suffix)
            model.eval()
            self.num_tasks = 1 if isinstance(model, NA2M) else model.num_tasks
            self.models.append(model)

        self._fitted = True
        return
    
class NA2MClassifier(NA2MBase):
    def __init__(
        self,
        units_multiplier: int = 2,
        num_basis_functions: int = 64,
        hidden_sizes: list = [64, 32],
        dropout: float = 0.1,
        feature_dropout: float = 0.05, 
        batch_size: int = 1024,
        num_workers: int = 0,
        num_epochs: int = 1000,
        log_dir: str = None,
        val_split: float = 0.15,
        device: str = 'cpu',
        lr: float = 0.02082,
        decay_rate: float = 0.0,
        output_reg: float = 0.2078,
        l2_reg: float = 0.0,
        save_model_frequency: int = 10,
        patience: int = 60,
        monitor_loss: bool = True,
        early_stop_mode: str = 'min',
        loss_func: Callable = None,
        metric: str = None,
        num_learners: int = 1,
        n_jobs: int = None,
        warm_start: bool = False,
        random_state: int = 42,
        interaction_indices: list = None,
    ) -> None:
        super(NA2MClassifier, self).__init__(
            units_multiplier=units_multiplier,
            num_basis_functions=num_basis_functions,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            feature_dropout=feature_dropout,
            batch_size=batch_size,
            num_workers=num_workers,
            num_epochs=num_epochs,
            log_dir=log_dir,
            val_split=val_split,
            device=device,
            lr=lr,
            decay_rate=decay_rate,
            output_reg=output_reg,
            l2_reg=l2_reg,
            save_model_frequency=save_model_frequency,
            patience=patience,
            monitor_loss=monitor_loss,
            early_stop_mode=early_stop_mode,
            loss_func=loss_func,
            metric=metric,
            num_learners=num_learners,
            n_jobs=n_jobs,
            warm_start=warm_start,
            random_state=random_state
        )
        self.regression = False
        self.interaction_indices = interaction_indices

    def fit(self, X, y, w=None):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()
        if isinstance(w, (pd.DataFrame, pd.Series)):
            w = w.to_numpy()
            
        if len(np.unique(y[~np.isnan(y)])) > 2:
            raise ValueError('More than two unique y-values detected. Multiclass classification not currently supported.')
        return super().fit(X, y, w)

    def _initialize_models(self, X, y):
        self.num_tasks = y.shape[1] if len(y.shape) > 1 else 1
        self.num_inputs = X.shape[1]
        self.models = []
        
        if self.interaction_indices is None:
            self.interaction_indices = [
                (i, j) for i in range(self.num_inputs) for j in range(i + 1, self.num_inputs)
            ]
        for _ in range(self.num_learners):
            model = NA2M(
                num_inputs=self.num_inputs,
                num_units=[64] * self.num_inputs,
                dropout=self.dropout,
                feature_dropout=self.feature_dropout,
                hidden_sizes=self.hidden_sizes,
                interaction_units=[64] * len(self.interaction_indices),
                interaction_indices=self.interaction_indices
            )
            self.models.append(model)

    def predict_proba(self, X) -> ArrayLike:
        out = scipy.special.expit(super().predict(X))
        return out

    def predict(self, X) -> ArrayLike:
        return self.predict_proba(X).round()
    
# Define NA2MRegressor class
class NA2MRegressor(NA2MBase):
    def __init__(
        self,
        units_multiplier: int = 2,
        num_basis_functions: int = 64,
        hidden_sizes: list = [64, 32],
        dropout: float = 0.1,
        feature_dropout: float = 0.05, 
        batch_size: int = 1024,
        num_workers: int = 0,
        num_epochs: int = 1000,
        log_dir: str = None,
        val_split: float = 0.15,
        device: str = 'cpu',
        lr: float = 0.02082,
        decay_rate: float = 0.0,
        output_reg: float = 0.2078,
        l2_reg: float = 0.0,
        save_model_frequency: int = 10,
        patience: int = 60,
        monitor_loss: bool = True,
        early_stop_mode: str = 'min',
        loss_func: Callable = None,
        metric: str = None,
        num_learners: int = 1,
        n_jobs: int = None,
        warm_start: bool = False,
        random_state: int = 42,
        interaction_indices: list = None,
    ) -> None:
        super(NA2MRegressor, self).__init__(
            units_multiplier=units_multiplier,
            num_basis_functions=num_basis_functions,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            feature_dropout=feature_dropout,
            batch_size=batch_size,
            num_workers=num_workers,
            num_epochs=num_epochs,
            log_dir=log_dir,
            val_split=val_split,
            device=device,
            lr=lr,
            decay_rate=decay_rate,
            output_reg=output_reg,
            l2_reg=l2_reg,
            save_model_frequency=save_model_frequency,
            patience=patience,
            monitor_loss=monitor_loss,
            early_stop_mode=early_stop_mode,
            loss_func=loss_func,
            metric=metric,
            num_learners=num_learners,
            n_jobs=n_jobs,
            warm_start=warm_start,
            random_state=random_state
        )
        self.regression = True
        self.interaction_indices = interaction_indices

    def _initialize_models(self, X, y):
        self.num_tasks = y.shape[1] if len(y.shape) > 1 else 1
        self.num_inputs = X.shape[1]
        self.models = []
        
        if self.interaction_indices is None:
            self.interaction_indices = [
                (i, j) for i in range(self.num_inputs) for j in range(i + 1, self.num_inputs)
            ]
        for _ in range(self.num_learners):
            model = NA2M(
                num_inputs=self.num_inputs,
                num_units=[64] * self.num_inputs,
                dropout=self.dropout,
                feature_dropout=self.feature_dropout,
                hidden_sizes=self.hidden_sizes,
                interaction_units=[64] * len(self.interaction_indices),
                interaction_indices=self.interaction_indices
            )
            self.models.append(model)

    def plot_interaction(self, feature_indices: Tuple[int, int]) -> None:
        num_samples = 100
        X = np.zeros((num_samples ** 2, self.num_inputs))
        grid_x, grid_y = np.meshgrid(
            np.linspace(-1.0, 1.0, num_samples),
            np.linspace(-1.0, 1.0, num_samples)
        )
        X[:, feature_indices[0]] = grid_x.ravel()
        X[:, feature_indices[1]] = grid_y.ravel()

        interaction_outputs = []
        for model in self.models:
            # Find the index of the interaction net corresponding to the given feature indices
            try:
                idx = model.interaction_indices.index(feature_indices)
            except ValueError:
                print(f"No interaction between features {feature_indices} in the model.")
                return
            interaction_nn = model.interaction_nns[idx]
            outputs = interaction_nn(torch.tensor(X, dtype=torch.float32))
            interaction_outputs.append(outputs.detach().cpu().numpy())

        # (learners, examples)
        interaction_outputs = np.stack(interaction_outputs, axis=0)
        # (examples,)
        y = np.mean(interaction_outputs, axis=0).squeeze()
        conf_int = np.std(interaction_outputs, axis=0).squeeze()

        # Reshape y and conf_int to grid
        y = y.reshape((num_samples, num_samples))
        conf_int = conf_int.reshape((num_samples, num_samples))
        return {'x': grid_x, 'y': grid_y, 'z': y, 'conf_int': conf_int}

# Utility function to compute the number of units
def get_num_units(
    units_multiplier: int,
    num_basis_functions: int,
    X: Union[np.ndarray, pd.DataFrame]
) -> List[int]:
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    num_unique_vals = [len(np.unique(X[:, i])) for i in range(X.shape[1])]
    num_units = [min(num_basis_functions, i * units_multiplier) for i in num_unique_vals]

    return num_units