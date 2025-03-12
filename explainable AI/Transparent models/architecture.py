import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Sequence
from typing import Tuple
from typing import Callable, List
from abc import abstractmethod
import sklearn.metrics as sk_metrics
import numpy as np
from itertools import combinations
from collections import defaultdict

class NBM(nn.Module):
    """
    Neural Basis Model where higher-order interactions of features are modeled
    as f(xi, xj) for order 2 or f(xi, xj, xk) for arbitrary order d.
    """

    def __init__(
        self,
        num_inputs: int,
        num_units: list,
        hidden_sizes: list,
        dropout: float,
        feature_dropout: float,
        nary_orders: List[int] = [1],
        num_bases: int = 100,
        hidden_dims: list = [256, 128, 128],
        interaction_indices: List[Tuple[int, ...]] = None,
        num_subnets: int = 1,
        batchnorm: bool = False,
        output_penalty: float = 0.0,
    ) -> None:
        super(NBM, self).__init__()
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.nary_orders = nary_orders
        self.num_bases = num_bases
        self.interaction_indices = interaction_indices
        self.hidden_dims = hidden_dims
        self.num_subnets = num_subnets
        self.batchnorm = batchnorm
        self.output_penalty = output_penalty

        # Build the n-ary indices
        self.nary_indices = {}
        
        if self.interaction_indices is not None:
            # 주어진 interaction_indices를 사용하여 nary_indices를 생성합니다.
            indices_by_order = defaultdict(list)
            for idx_tuple in self.interaction_indices:
                order = len(idx_tuple)
                indices_by_order[str(order)].append(idx_tuple)
            self.nary_indices = dict(indices_by_order)
        else:
            
            for order in self.nary_orders:
                self.nary_indices[str(order)] = list(combinations(range(self.num_inputs), order))

        # Create the bases_nary_models
        self.bases_nary_models = nn.ModuleDict()
        for order in self.nary_indices.keys():
            for subnet in range(self.num_subnets):
                key = f'ord{order}_net{subnet}'
                self.bases_nary_models[key] = ConceptNNBasesNary(
                    order=int(order),
                    num_bases=self.num_bases,
                    hidden_dims=self.hidden_dims,
                    dropout=self.dropout,
                    batchnorm=self.batchnorm,
                )

        self.bases_dropout = nn.Dropout(p=self.feature_dropout)

        num_out_features = sum(len(self.nary_indices[order]) for order in self.nary_indices.keys()) * self.num_subnets

        self.featurizer = nn.Conv1d(
            in_channels=num_out_features * self.num_bases,
            out_channels=num_out_features,
            kernel_size=1,
            groups=num_out_features,
        )

        self.classifier = nn.Linear(
            in_features=num_out_features,
            out_features=1,
            bias=True,
        )

        self._bias = torch.nn.Parameter(data=torch.zeros(1))

    def get_key(self, order, subnet):
        return f'ord{order}_net{subnet}'

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bases = []
        for order in self.nary_indices.keys():
            for subnet in range(self.num_subnets):
                key = self.get_key(order, subnet)
                indices = self.nary_indices[order]
                input_order = inputs[:, indices]

                # Reshape input_order to (batch_size * num_combinations, order)
                batch_size = inputs.size(0)
                num_combinations = len(indices)
                input_order = input_order.view(batch_size * num_combinations, -1)

                # Pass through the bases_nary_models
                bases_out = self.bases_dropout(self.bases_nary_models[key](input_order))
                # Reshape back to (batch_size, num_combinations, num_bases)
                bases_out = bases_out.view(batch_size, num_combinations, -1)
                bases.append(bases_out)

        bases = torch.cat(bases, dim=1)  # Concatenate along num_combinations dimension

        # Featurizer
        bases = bases.view(inputs.size(0), -1, 1)  # Reshape to (batch_size, num_features, 1)
        out_feats = self.featurizer(bases).squeeze(-1)  # Shape: (batch_size, num_out_features)

        # Classifier
        out = self.classifier(out_feats)
        return out.squeeze(-1) + self._bias, out_feats
    
# Define NAM class with interaction terms
class NA2M(nn.Module):

    def __init__(
        self,
        num_inputs: int,
        num_units: list,
        hidden_sizes: list,
        dropout: float,
        feature_dropout: float,
        interaction_units: list = None,
        interaction_indices: list = None,
    ) -> None:
        super(NA2M, self).__init__()
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.feature_dropout = feature_dropout

        self.dropout_layer = nn.Dropout(p=self.feature_dropout)

        ## Builds the FeatureNNs
        self.feature_nns = nn.ModuleList([
            FeatureNN(
                input_shape=1, 
                num_units=64,
                dropout=self.dropout, feature_num=i, 
                hidden_sizes=self.hidden_sizes
            )
            for i in range(num_inputs)
        ])

        # Build InteractionNNs
        if interaction_units is None:
            interaction_units = self.num_units

        if interaction_indices is None:
            # By default, include all pairs of features
            interaction_indices = [(i, j) for i in range(num_inputs) for j in range(i+1, num_inputs)]

        self.interaction_indices = interaction_indices

        self.interaction_nns = nn.ModuleList([
            InteractionNN(
                feature_indices=pair,
                num_units=64,
                dropout=dropout,
                hidden_sizes=hidden_sizes
            )
            for pair in self.interaction_indices
        ])

        self._bias = nn.Parameter(torch.zeros(1))

    def calc_outputs(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """Returns the output computed by each feature net and interaction net."""
        # Outputs from feature_nns
        feature_outputs = [self.feature_nns[i](inputs[:, i]) for i in range(self.num_inputs)]
        # Outputs from interaction_nns
        interaction_outputs = [interaction_nn(inputs) for interaction_nn in self.interaction_nns]
        return feature_outputs + interaction_outputs

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        individual_outputs = self.calc_outputs(inputs)
        conc_out = torch.cat(individual_outputs, dim=-1)
        dropout_out = self.dropout_layer(conc_out)

        out = torch.sum(dropout_out, dim=-1)
        return out + self._bias, dropout_out



class NAM(torch.nn.Module):

    def __init__(
        self,
        num_inputs: int,
        num_units: list,
        hidden_sizes: list,
        dropout: float,
        feature_dropout: float
    ) -> None:
        super(NAM, self).__init__()
        assert len(num_units) == num_inputs
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.feature_dropout = feature_dropout

        self.dropout_layer = nn.Dropout(p=self.feature_dropout)

        ## Builds the FeatureNNs on the first call.
        self.feature_nns = nn.ModuleList([
            FeatureNN(
                input_shape=1, 
                num_units=64,
                dropout=self.dropout, feature_num=i, 
                hidden_sizes=self.hidden_sizes
            )
            for i in range(num_inputs)
        ])

        self._bias = torch.nn.Parameter(data=torch.zeros(1))

    def calc_outputs(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """Returns the output computed by each feature net."""
        return [self.feature_nns[i](inputs[:, i]) for i in range(self.num_inputs)]

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        individual_outputs = self.calc_outputs(inputs)
        conc_out = torch.cat(individual_outputs, dim=-1)
        dropout_out = self.dropout_layer(conc_out)

        out = torch.sum(dropout_out, dim=-1)
        return out + self._bias, dropout_out

class FeatureNN(torch.nn.Module):
    """Neural Network model for each individual feature."""

    def __init__(
        self,
        input_shape: int,
        feature_num: int,
        num_units: int,
        dropout: float,
        hidden_sizes: list = [64, 32],
        activation: str = 'relu'
    ) -> None:
        """Initializes FeatureNN hyperparameters.

        Args:
          num_units: Number of hidden units in first hidden layer.
          dropout: Coefficient for dropout regularization.
          feature_num: Feature Index used for naming the hidden layers.
        """
        super(FeatureNN, self).__init__()
        self._input_shape = input_shape
        self._num_units = num_units
        self._feature_num = feature_num
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        
        all_hidden_sizes = [self._num_units] + self._hidden_sizes

        layers = []

        self.dropout = nn.Dropout(p=dropout)

        ## First layer is ExU
        if self._activation == "exu":
            layers.append(ExU(in_features=input_shape, out_features=num_units))
        else:
            layers.append(LinReLU(in_features=input_shape, out_features=num_units))

        ## Hidden Layers
        for in_features, out_features in zip(all_hidden_sizes, all_hidden_sizes[1:]):
            layers.append(LinReLU(in_features, out_features))

        ## Last Linear Layer
        layers.append(nn.Linear(in_features=all_hidden_sizes[-1], out_features=1, bias=False))

        self.model = nn.ModuleList(layers)

    def forward(self, inputs) -> torch.Tensor:
        """Computes FeatureNN output with either evaluation or training
        mode."""
        outputs = inputs.unsqueeze(1)
        for layer in self.model:
            outputs = self.dropout(layer(outputs))
        return outputs
    
class InteractionNN(nn.Module):
    """Neural Network model for interaction between two features."""
    
    def __init__(
        self,
        feature_indices: Tuple[int, int],
        num_units: int,
        dropout: float,
        hidden_sizes: list = [64, 32],
        activation: str = 'relu'
    ) -> None:
        super(InteractionNN, self).__init__()
        self.feature_indices = feature_indices
        self.num_units = num_units
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        all_hidden_sizes = [self.num_units] + self.hidden_sizes

        layers = []

        self.dropout = nn.Dropout(p=dropout)

        ## First layer
        input_dim = 2  # since we're using two features
        if self.activation == "exu":
            layers.append(ExU(in_features=input_dim, out_features=num_units))
        else:
            layers.append(LinReLU(in_features=input_dim, out_features=num_units))

        ## Hidden Layers
        for in_features, out_features in zip(all_hidden_sizes, all_hidden_sizes[1:]):
            layers.append(LinReLU(in_features, out_features))

        ## Last Linear Layer
        layers.append(nn.Linear(in_features=all_hidden_sizes[-1], out_features=1, bias=False))

        self.model = nn.ModuleList(layers)

    def forward(self, inputs) -> torch.Tensor:
        """Computes InteractionNN output."""
        # Extract the two features
        feature_1 = inputs[:, self.feature_indices[0]].unsqueeze(1)
        feature_2 = inputs[:, self.feature_indices[1]].unsqueeze(1)
        features = torch.cat([feature_1, feature_2], dim=1)
        outputs = features
        for layer in self.model:
            outputs = self.dropout(layer(outputs))
        return outputs

    def extra_repr(self):
        return f'feature_indices={self.feature_indices}, num_units={self.num_units}'

class ConceptNNBasesNary(nn.Module):
    """Neural Network learning bases."""

    def __init__(
        self, order, num_bases, hidden_dims, dropout=0.0, batchnorm=False
    ) -> None:
        """Initializes ConceptNNBases hyperparameters.
        Args:
            order: Order of N-ary concept interactions.
            num_bases: Number of bases learned.
            hidden_dims: Number of units in hidden layers.
            dropout: Coefficient for dropout regularization.
            batchnorm (True): Whether to use batchnorm or not.
        """
        super(ConceptNNBasesNary, self).__init__()

        assert order > 0, "Order of N-ary interactions has to be larger than '0'."

        layers = []
        self._model_depth = len(hidden_dims) + 1
        self._batchnorm = batchnorm

        # First input_dim depends on the N-ary order
        input_dim = order
        for dim in hidden_dims:
            layers.append(nn.Linear(in_features=input_dim, out_features=dim))
            if self._batchnorm is True:
                layers.append(nn.BatchNorm1d(dim))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.ReLU())
            input_dim = dim

        # Last MLP layer
        layers.append(nn.Linear(in_features=input_dim, out_features=num_bases))
        # Add batchnorm and relu for bases
        if self._batchnorm is True:
            layers.append(nn.BatchNorm1d(num_bases))
        layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class LinReLU(torch.nn.Module):
    __constants__ = ['bias']

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super(LinReLU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weights)
        torch.nn.init.trunc_normal_(self.bias, std=0.5)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        output = (inputs - self.bias) @ self.weights
        output = F.relu(output)

        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'
    
class ExU(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super(ExU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        ## Page(4): initializing the weights using a normal distribution
        ##          N(x; 0:5) with x 2 [3; 4] works well in practice.
        torch.nn.init.trunc_normal_(self.weights, mean=4.0, std=0.5)
        torch.nn.init.trunc_normal_(self.bias, std=0.5)

    def forward(
        self,
        inputs: torch.Tensor,
        n: int = 1,
    ) -> torch.Tensor:
        output = (inputs - self.bias).matmul(torch.exp(self.weights))

        # ReLU activations capped at n (ReLU-n)
        output = F.relu(output)
        output = torch.clamp(output, 0, n)

        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'

