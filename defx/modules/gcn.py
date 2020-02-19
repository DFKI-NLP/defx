from typing import List, Optional

import math
import torch
from allennlp.modules import Seq2SeqEncoder
from torch import nn
from overrides import overrides
from allennlp.nn import Activation


class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Code taken from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True) -> None:
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self,
                inputs: torch.Tensor,
                adjacency: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        batch_size, seq_len, _ = inputs.size()

        inputs = inputs.view(batch_size * seq_len, -1)

        support = torch.mm(inputs, self.weight)
        support = support.view(batch_size, seq_len, -1)

        output = torch.bmm(adjacency, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


@Seq2SeqEncoder.register("gcn")
class GCN(Seq2SeqEncoder):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float = 0,
                 gcn_layer_activation: Optional[Activation] = None):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._activation = gcn_layer_activation or Activation.by_name("relu")()

        self._gcn_layers = []  # type: List[GraphConvolution]

        gcn_input_size = input_size
        for layer_idx in range(num_layers):
            gcn_layer = GraphConvolution(in_features=gcn_input_size,
                                         out_features=hidden_size)
            self.add_module(f"gcn_layer_{layer_idx}", gcn_layer)
            self._gcn_layers.append(gcn_layer)
            gcn_input_size = hidden_size

        self.dropout = torch.nn.Dropout(dropout)

    @overrides
    def get_input_dim(self) -> int:
        return self.input_size

    @overrides
    def get_output_dim(self) -> int:
        return self.hidden_size

    @overrides
    def is_bidirectional(self) -> bool:
        return False

    def forward(self,
                inputs: torch.Tensor,
                mask: torch.Tensor,
                adjacency: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        if mask is not None:
            inputs = inputs * mask.unsqueeze(-1).float()

        denom = adjacency.sum(dim=2, keepdim=True) + 1

        output = inputs
        for i in range(len(self._gcn_layers)):
            gcn_layer_i = getattr(self, f"gcn_layer_{i}")
            output = self._activation(gcn_layer_i(output, adjacency) / denom)
            if i < len(self._gcn_layers) - 1:
                output = self.dropout(output)

        return output
