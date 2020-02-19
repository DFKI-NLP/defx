from typing import Optional

import torch
from allennlp.modules import Seq2SeqEncoder
from overrides import overrides

from defx.modules.gcn import GCN


@Seq2SeqEncoder.register("stacked-seq-encoders")
class StackedSeqEncoders(Seq2SeqEncoder):

    def __init__(self,
                 base_encoder: Seq2SeqEncoder,
                 top_encoder: Seq2SeqEncoder,
                 base_encoding_dropout: float = None):
        super().__init__()

        assert top_encoder.get_input_dim() == base_encoder.get_output_dim(),\
            "stacked encoder dimension must match"

        self._base_encoder = base_encoder
        self._top_encoder = top_encoder
        if base_encoding_dropout:
            self._base_dropout = torch.nn.Dropout(base_encoding_dropout)
        else:
            self._base_dropout = lambda x: x

    @overrides
    def get_input_dim(self) -> int:
        return self._base_encoder.get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return self._top_encoder.get_output_dim()

    @overrides
    def is_bidirectional(self) -> bool:
        return self._top_encoder.is_bidirectional()

    def forward(self,
                tokens: torch.Tensor,
                mask: torch.Tensor,
                adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:

        if isinstance(self._base_encoder, GCN):
            output = self._base_encoder(tokens, mask, adjacency)
        else:
            output = self._base_encoder(tokens, mask)
        output = self._base_dropout(output)

        if isinstance(self._top_encoder, GCN):
            return self._top_encoder(output, mask, adjacency)
        else:
            return self._top_encoder(output, mask)

