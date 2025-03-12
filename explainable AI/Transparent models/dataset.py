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

class NAMDataset(torch.utils.data.Dataset):

    def __init__(self,
                 X: Union[ArrayLike, pd.DataFrame],
                 y: Union[ArrayLike, pd.DataFrame],
                 w: Union[ArrayLike, pd.DataFrame] = None):
        """Dataset for NAMs that handles default weights.

        Args:
            X (Union[ArrayLike, pd.DataFrame]): Feature array.
            y (Union[ArrayLike, pd.DataFrame]): Target array.
            w (Union[ArrayLike, pd.DataFrame]): Weight array.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()
            
        self.X = torch.tensor(X, requires_grad=False, dtype=torch.float)
        self.y = torch.tensor(y, requires_grad=False, dtype=torch.float)

        if not w:
            self.w = torch.clone(self.y)
            self.w[~torch.isnan(self.w)] = 1.0
            self.w[torch.isnan(self.w)] = 0.0
        else:
            self.w = torch.tensor(w, requires_grad=False, dtype=torch.float)

        if len(self.y.shape) > 1:
            # In multitask setting, set missing labels to 0. The loss
            # contributions from these examples will get zeroed out downstream
            # but leaving nan values will cause a crash.
            self.y[self.y != self.y] = 0.0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[ArrayLike, ...]:
        return self.X[idx], self.y[idx], self.w[idx]