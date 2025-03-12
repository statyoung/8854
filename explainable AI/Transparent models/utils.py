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


def weighted_loss(loss_func: Callable, logits: torch.Tensor, targets: torch.Tensor, weights: torch.tensor) -> torch.Tensor:
    loss = loss_func(logits, targets, reduction='none')
    loss *= weights
    loss = torch.sum(loss, dim=0)
    loss = loss / torch.sum(weights, dim=0)
    return torch.mean(loss)


def reg_penalty(fnn_out: torch.Tensor, model: nn.Module,
    output_regularization: float, l2_regularization: float
) -> torch.Tensor:
    """Computes penalized loss with L2 regularization and output penalty.

    Args:
      config: Global config.
      model: Neural network model.
      inputs: Input values to be fed into the model for computing predictions.
      targets: Target values containing either real values or binary labels.

    Returns:
      The penalized loss.
    """

    def features_loss(per_feature_outputs):
        b, f = per_feature_outputs.shape[0], per_feature_outputs.shape[-1]
        out = torch.sum(per_feature_outputs ** 2) / (b * f)

        return output_regularization * out

    def weight_decay(model: nn.Module) -> torch.Tensor:
        """Penalizes the L2 norm of weights in each feature net."""
        num_networks = len(model.feature_nns)
        l2_losses = [(x**2).sum() for x in model.parameters()]
        return sum(l2_losses) / num_networks

    reg_loss = 0.0
    if output_regularization > 0:
        reg_loss += features_loss(fnn_out)

    if l2_regularization > 0:
        reg_loss += l2_regularization * weight_decay(model)

    return reg_loss


def make_penalized_loss_func(loss_func, regression, output_regularization, l2_regularization):
    def penalized_loss_func(logits, targets, weights, fnn_out, model):
        loss = weighted_loss(loss_func, logits, targets, weights)
        loss += reg_penalty(fnn_out, model, output_regularization, l2_regularization)
        return loss

    if not loss_func:
        loss_func = F.mse_loss if regression else F.binary_cross_entropy_with_logits
    return penalized_loss_func

class Metric:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class Accuracy(Metric):
    def __init__(
        self, 
        input_type: str = None
    ) -> None:
        self._num = 0
        self._denom = 0
        self._input_type = input_type
        self._updated = False

    def update(self, predictions, targets) -> None:
        # TODO: Exception handling/input checking
        predictions, targets = predictions.detach(), targets.detach()
        if self._input_type == 'scores':
            predictions = predictions.round()
        elif self._input_type == 'logits':
            predictions = torch.sigmoid(predictions).round()

        self._num += (predictions * targets).sum().item()
        self._denom += len(predictions)
        self._updated = True
        return

    def compute(self) -> None:
        if not self._updated:
            # TODO: Find appropriate exception
            raise Exception()
        
        return self._num / self._denom

    def reset(self) -> None:
        self._num = self._denom = 0
        return


class AUC(Metric):
    def __init__(
        self 
    ) -> None:
        self._predictions = []
        self._targets = []
        self._updated = False

    @abstractmethod
    def score_function(self, predictions, targets) -> float:
        pass

    def update(self, predictions, targets) -> None:
        # TODO: Exception handling/input checking
        self._predictions.append(predictions)
        self._targets.append(targets)
        self._updated = True
        return

    def compute(self) -> None:
        if not self._updated:
            # TODO: Find appropriate exception
            raise Exception()
        
        predictions = torch.cat(self._predictions).detach().cpu().numpy()
        targets = torch.cat(self._targets).detach().cpu().numpy()
        return self.score_function(predictions, targets)

    def reset(self) -> None:
        self._predictions = []
        self._targets = []
        return


class AUROC(AUC):
    def __init__(
        self 
    ) -> None:
        super(AUROC, self).__init__()
        return

    def score_function(self, predictions, targets) -> float:
        return sk_metrics.roc_auc_score(targets, predictions)


class AveragePrecision(AUC):
    def __init__(
        self 
    ) -> None:
        super(AveragePrecision, self).__init__()
        return

    def score_function(self, predictions, targets) -> float:
        return sk_metrics.average_precision_score(targets, predictions)


class MeanError(Metric):
    def __init__(
        self 
    ) -> None:
        self._sum_of_errors = 0
        self._num_examples = 0
        self._updated = True
        return

    @abstractmethod
    def distance_func(self, predictions, targets) -> float:
        pass

    def update(self, predictions, targets) -> None:
        # TODO: Exception handling/input checking
        predictions = predictions.detach().cpu().numpy() 
        targets = targets.detach().cpu().numpy()
        self._sum_of_errors += self.distance_func(predictions, targets)
        self._num_examples += predictions.shape[0]
        self._updated = True
        return

    def compute(self) -> None:
        if not self._updated:
            # TODO: Find appropriate exception
            raise Exception()
        
        return self._sum_of_errors / self._num_examples

    def reset(self) -> None:
        self._sum_of_errors = 0
        self._num_examples = 0
        return


class MeanSquaredError(MeanError):
    def __init__(
        self 
    ) -> None:
        super(MeanSquaredError, self).__init__()
        return

    def distance_func(self, predictions, targets) -> float:
        return np.sum((predictions - targets) ** 2)


class RootMeanSquaredError(MeanSquaredError):
    def __init__(
        self 
    ) -> None:
        super(MeanSquaredError, self).__init__()
        return

    def distance_func(self, predictions, targets) -> float:
        return super().distance_func(predictions, targets) ** 0.5


class MeanAbsoluteError(MeanError):
    def __init__(
        self 
    ) -> None:
        super(MeanSquaredError, self).__init__()
        return

    def distance_func(self, predictions, targets) -> float:
        return np.sum(np.absolute(predictions - targets))
    
class Checkpointer:
    """A simple PyTorch model load/save wrapper."""

    def __init__(
        self,
        log_dir: str = 'output',
        device: str = 'cpu',
        random_state: int = None,  # 시드를 추가로 받음
    ) -> None:
        """Constructs a simple load/save checkpointer."""
        if random_state is not None:
            # 랜덤 시드를 디렉토리 이름에 추가
            log_dir = os.path.join(log_dir, f"seed_{random_state}")
        self._ckpt_dir = os.path.join(log_dir, "ckpts")
        self._device = device
        os.makedirs(self._ckpt_dir, exist_ok=True)

    def save(
        self,
        model,
        epoch: int,
    ) -> str:
        """Saves the model to the ckpt_dir/epoch/model.pt file."""
        ckpt_path = os.path.join(self._ckpt_dir, "model-{}.pt".format(epoch))
        torch.save({
            'model_state_dict': model.state_dict(),
            'attributes': vars(model),
            'class': type(model)
            }, 
            ckpt_path
        )
        return ckpt_path

    def load(
        self,
        epoch: int,
    ) -> nn.Module:
        """Loads the model from the ckpt_dir/epoch/model.pt file."""
        ckpt_path = os.path.join(self._ckpt_dir, "model-{}.pt".format(epoch))
        ckpt = torch.load(ckpt_path, map_location=self._device)
        constructor = ckpt['class']
        constructor_args = inspect.getfullargspec(constructor).args
        args = {k: v for k, v in ckpt['attributes'].items() if k in constructor_args}
        model = constructor(**args)
        model.load_state_dict(ckpt['model_state_dict'])
        return model

# 정의 나중에 적기
def get_num_units(
    units_multiplier: int,
    num_basis_functions: int,
    X: Union[ArrayLike, pd.DataFrame]
) -> List:
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    num_unique_vals = [len(np.unique(X[:, i])) for i in range(X.shape[1])]
    num_units = [min(num_basis_functions, i * units_multiplier) for i in num_unique_vals]

    return num_units