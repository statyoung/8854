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
from src.nam_architecture import *
from src.nam2_architecture import *
from src.nbm_architecture import *


if __name__ == '__main__':
    
    def create_parser():
        parser = argparse.ArgumentParser(description='XAI')
        parser.add_argument('--seed', default=0, type=int, help='seed number')
        parser.add_argument('--gpu', default='cpu', type=str, help = 'gpu number')
        parser.add_argument('--model', default='nam', type=str, choices=['nam', 'nam2', 'nbm', 'nbm2','namc','nam2c','nbmc','nbm2c'])
        return parser
    
    args = create_parser().parse_args()
    for key, value in vars(args).items():
        print(f'\t [{key}]: {value}')
    
    # device 및 저장경로 정의
    device = torch.device(f'cuda:{args.gpu}') if (torch.cuda.is_available() & (args.gpu != 'cpu')) else torch.device('cpu')
    output_dir = f'{args.model}/output'
    
   
    column_names = [
    "longitude", "latitude", "housingMedianAge", "totalRooms",
    "totalBedrooms", "population", "households", "medianIncome",
    "medianHouseValue"
    ]
    data = pd.read_csv("cal_housing.data", header=None, names=column_names)
    data['medianHouseValue'] = data['medianHouseValue'] / 100000
    X = data.drop(columns=['medianHouseValue'])  
    y = data['medianHouseValue']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_processed = pd.DataFrame(X_scaled, columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=args.seed)
    
    if args.model == 'nam':
        model = NAMRegressor(
            num_epochs=1000,
            lr = 0.00674,
            output_reg= 0.001,
            dropout= 0.0,
            feature_dropout= 0.0,
            decay_rate=0.000001,
            num_learners=1,
            patience=0,
            metric='rmse',
            n_jobs=None,
            log_dir = output_dir,
            device = device,
            random_state=args.seed,
        )
    elif args.model == 'nam2':
        model = NA2MRegressor(
            num_epochs=1000,
            lr = 0.00674,
            output_reg= 0.001,
            dropout= 0.0,
            feature_dropout= 0.0,
            decay_rate= 0.000001,
            num_learners=1,
            patience=0,
            metric='rmse',
            n_jobs=None,
            log_dir = output_dir,
            device = device,
            random_state=args.seed,
        )
    elif args.model == 'nbm':
        model = NBMRegressor(
            nary_orders=[1],
            num_learners=1,
            lr = 0.00197,
            decay_rate= 0.00001568,
            dropout=0,
            feature_dropout=0.05,
            num_bases=100,
            hidden_dims=[256,128,128],
            patience=0,
            metric='rmse',
            n_jobs=None,
            log_dir = output_dir,
            device = device,
            random_state=args.seed
        )
    elif args.model == 'nbm2':
        model = NBMRegressor(
            nary_orders=[1,2],
            num_learners=1,
            lr = 0.00190,
            decay_rate= 0.000000007483,
            dropout=0,
            feature_dropout=0.05,
            num_bases=200,
            hidden_dims=[256,128,128],
            patience=0,
            metric='rmse',
            n_jobs=None,
            log_dir = output_dir,
            device = device,
            random_state=args.seed
        )
    elif args.model == 'namc':
        model = NAMClassifier(
            num_epochs=1000,
            num_learners=1,
            patience=0,
            metric='AUROC',
            n_jobs=None,
            log_dir = output_dir,
            device = device,
            random_state=args.seed
        )
    elif args.model == 'nam2c':
        model = NA2MClassifier(
            num_epochs=1000,
            num_learners=1,
            patience=0,
            metric='AUROC',
            n_jobs=None,
            log_dir = output_dir,
            device = device,
            random_state=args.seed
        )
    elif args.model == 'nbmc':
        model = NBMClassifier(
            nary_orders=[1],
            num_learners=1,
            num_bases=100,
            hidden_dims=[256,128,128],
            patience=0,
            metric='AUROC',
            n_jobs=None,
            log_dir = output_dir,
            device = device,
            random_state=args.seed
        )
    elif args.model == 'nbm2c':
        model = NBMClassifier(
            nary_orders=[1,2],
            num_learners=1,
            num_bases=200,
            hidden_dims=[256,128,128],
            patience=0,
            metric='AUROC',
            n_jobs=None,
            log_dir = output_dir,
            device = device,
            random_state=args.seed
        )

    model.fit(X_train, y_train)
