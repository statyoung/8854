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

class NBMBase:
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
        interaction_indices: List[Tuple[int, int]] = None,
        monitor_loss: bool = True,
        early_stop_mode: str = 'min',
        loss_func: Callable = None,
        metric: str = None,
        num_learners: int = 1,
        n_jobs: int = None,
        warm_start: bool = False,
        random_state: int = 42,
        nary_orders: List[int] = [1],
        num_bases: int = 100,
        hidden_dims: list = [256, 128, 128],
        num_subnets: int = 1,
        batchnorm: bool = False,
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
        self.interaction_indices = interaction_indices
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

        self.nary_orders = nary_orders
        self.num_bases = num_bases
        self.hidden_dims = hidden_dims
        self.num_subnets = num_subnets
        self.batchnorm = batchnorm

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
            model = NBM(
                num_inputs=self.num_inputs,
                num_units=get_num_units(self.units_multiplier, self.num_basis_functions, X),
                dropout=self.dropout,
                feature_dropout=self.feature_dropout,
                hidden_sizes=self.hidden_sizes,
                nary_orders=self.nary_orders,
                num_bases=self.num_bases,
                hidden_dims=self.hidden_dims,
                num_subnets=self.num_subnets,
                batchnorm=self.batchnorm,
                output_penalty=self.output_reg,
                interaction_indices=self.interaction_indices
            )
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
        if isinstance(w, (pd.DataFrame, pd.Series)):
            w = w.to_numpy()

        self._set_random_state()
        if not self.warm_start or not self._fitted:
            self._initialize_models(X, y)

        self.partial_fit(X, y, w)
        return self

    def partial_fit(self, X, y, w=None) -> None:
        self._models_to_device(self.device)
        
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

    def predict(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = torch.tensor(X, requires_grad=False, dtype=torch.float)
        predictions = np.zeros((X.shape[0],))
        if self.num_tasks > 1:
            predictions = np.zeros((X.shape[0], self.num_tasks))

        for model in self.models:
            preds, _ = model.forward(X)
            predictions += preds.detach().cpu().numpy()

        return predictions / self.num_learners

    def load_checkpoints(self, checkpoint_dir):
        self.models = []
        for i in range(self.num_learners):
            checkpointer = Checkpointer(os.path.join(checkpoint_dir, str(i)))
            model = checkpointer.load(self._best_checkpoint_suffix)
            model.eval()
            self.num_tasks = 1
            self.models.append(model)

        self._fitted = True
        return

class NBMClassifier(NBMBase):
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
        nary_orders: List[int] = [1],
        num_bases: int = 100,
        hidden_dims: list = [256, 128, 128],
        num_subnets: int = 1,
        batchnorm: bool = False,
    ) -> None:
        super(NBMClassifier, self).__init__(
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
            random_state=random_state,
            nary_orders=nary_orders,
            num_bases=num_bases,
            hidden_dims=hidden_dims,
            num_subnets=num_subnets,
            batchnorm=batchnorm,
        )
        self.regression = False  # classifier
    def fit(self, X, y, w=None):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            w = w.to_numpy()
            
        if len(np.unique(y[~np.isnan(y)])) > 2:
            raise ValueError('More than two unique y-values detected. Multiclass classification not currently supported.')
        return super().fit(X, y, w)

    def predict_proba(self, X) -> ArrayLike:
        out = scipy.special.expit(super().predict(X))
        return out

    def predict(self, X) -> ArrayLike:
        return self.predict_proba(X).round()
    
class NBMRegressor(NBMBase):
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
        interaction_indices: List[Tuple[int, int]]=None,
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
        nary_orders: List[int] = [1],
        num_bases: int = 100,
        hidden_dims: list = [256, 128, 128],
        num_subnets: int = 1,
        batchnorm: bool = False,
    ) -> None:
        super(NBMRegressor, self).__init__(
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
            interaction_indices=interaction_indices,
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
            random_state=random_state,
            nary_orders=nary_orders,
            num_bases=num_bases,
            hidden_dims=hidden_dims,
            num_subnets=num_subnets,
            batchnorm=batchnorm,
        )
        self.regression = True