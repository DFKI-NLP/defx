from typing import Optional, List, Dict

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
from torch import nn
from torch.nn import Linear, Parameter

from defx.models import RelationScorer


@RelationScorer.register('auxiliary_relation_scorer')
class AuxiliaryRelationScorer(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 input_size: int,
                 hidden_size: int,
                 label_namespace: str = "labels",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self._idx_to_label = self.vocab.get_index_to_token_vocabulary(
            label_namespace
        )
        num_labels = self.vocab.get_vocab_size(label_namespace)

        # projection layers
        self._U = Linear(input_size, hidden_size, bias=False)
        self._W = Linear(input_size, hidden_size, bias=False)
        self._V = Linear(hidden_size, num_labels, bias=False)
        self._b = Parameter(torch.Tensor(hidden_size))
        nn.init.normal_(self._b)

        self._accuracy = CategoricalAccuracy()

        initializer(self)


    @overrides
    def forward(self,
                sequence: torch.LongTensor,
                mask: torch.LongTensor,
                relation_root_idxs: torch.LongTensor = None,
                relations: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        # pylint: disable=no-member

        assert relation_root_idxs is None, 'This implementation does not take relation idxs'

        # Shape: batch, seq_len, input_size
        batch_size, seq_len, _ = sequence.size()

        head = self._U(sequence)
        tail = self._W(sequence)

        head = head.unsqueeze(2)  # Shape: batch, seq_len, 1, hidden_size
        tail = tail.unsqueeze(1)  # Shape: batch, 1, seq_len, hidden_size

        # Shape: batch x seq_len x seq_len x hidden_size
        head_tail = head + tail + self._b  # cross-product sum
        activations = F.relu(head_tail)

        # Shape: batch x seq_len x (seq_len * num_classes)
        logits = self._V(activations)

        # Shape: batch x seq_len x (seq_len * num_classes) + 1
        relation_scores = torch.softmax(logits, dim=-1)

        output_dict = {'logits': logits, 'relation_scores': relation_scores}

        if relations is not None:
            relations_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
            self._accuracy(logits, relations, relations_mask)
            unreduced_loss = F.cross_entropy(logits.permute(0, 3, 1, 2), relations, reduction='none')
            masked_loss = unreduced_loss * relations_mask.float()
            output_dict["loss"] = masked_loss.mean()

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            're_acc': self._accuracy.get_metric(reset=reset),
        }
