from typing import Optional, List, Dict

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
from torch import nn
from torch.nn import Linear, Parameter

from defx.metrics import F1Measure
from defx.util.index_to_relation_and_type_mapping import map_relation_head_and_type_to_index, \
    map_index_to_relation_head_and_type


@Model.register('relation_scorer')
class RelationScorer(Model):
    """
    Takes a sequence of encoded tokens and scores all existing relations.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    input_size : ``int``, required.
        The dimension of the encoded input sequence.
    hidden_size : ``int``, required.
        The dimension of the hidden internal representations.
    label_namespace: ``str``, optional(default = "labels")
        Vocabulary namespace corresponding to labels.
        By default, we use the "labels" namespace.
    negative_label: ``str``, optional(default = "0")
        The label of the negative relation class.
    verbose_metrics: ``bool``, optional(default = False)
        Enable to print detailed per-class metrics.
    ignored_labels: ``List[str]``, optional(default = [negative_label])
        Relation labels to ignore in the f1 score evaluation, defaults to the negative class only.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 input_size: int,
                 hidden_size: int,
                 label_namespace: str = "labels",
                 negative_label: str = "0",
                 verbose_metrics: bool = False,
                 ignored_labels: List[str] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self._idx_to_label = self.vocab.get_index_to_token_vocabulary(
            label_namespace
        )
        self._num_pos_classes = self.vocab.get_vocab_size(label_namespace) - 1
        self._negative_label_idx = self.vocab.get_token_index(negative_label,
                                                              namespace=label_namespace)
        assert self._negative_label_idx == 0, "negative label idx is supposed to be 0"

        # projection layers
        self._U = Linear(input_size, hidden_size, bias=False)
        self._W = Linear(input_size, hidden_size, bias=False)
        self._V = Linear(hidden_size, self._num_pos_classes, bias=False)
        self._T = Linear(hidden_size, 1, bias=False)
        self._b = Parameter(torch.Tensor(hidden_size))
        nn.init.normal_(self._b)

        self.metrics = {"accuracy": CategoricalAccuracy()}
        self._verbose_metrics = verbose_metrics
        if not ignored_labels:
            ignored_labels = [negative_label]
        else:
            if negative_label not in ignored_labels:
                ignored_labels.append(negative_label)
        self._f1_metric = F1Measure(vocabulary=vocab,
                                    negative_label=negative_label,
                                    average='macro',
                                    label_namespace=label_namespace,
                                    ignored_labels=ignored_labels)

        initializer(self)

    @overrides
    def forward(self,
                sequence: torch.LongTensor,
                mask: torch.LongTensor,
                relation_root_idxs: torch.LongTensor = None,
                relations: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        # pylint: disable=no-member

        # Shape: batch, seq_len, input_size
        batch_size, seq_len, _ = sequence.size()

        head = self._U(sequence)
        tail = self._W(sequence)

        head = head.unsqueeze(2)  # Shape: batch, seq_len, 1, hidden_size
        tail = tail.unsqueeze(1)  # Shape: batch, 1, seq_len, hidden_size

        # Shape: batch x seq_len x seq_len x hidden_size
        head_tail = head + tail + self._b  # cross-product sum
        activations = F.relu(head_tail)

        # Shape: batch x seq_len x hidden_size
        activations_agg, _ = activations.max(dim=2, keepdim=False)

        # Shape: batch x seq_len x 1
        negative_logits = self._T(activations_agg)

        # Shape: batch x seq_len x (seq_len * num_classes)
        positive_logits = self._V(activations)
        positive_logits = positive_logits.view(
            [batch_size, seq_len, seq_len * self._num_pos_classes]
        )

        # Shape: batch x seq_len x (seq_len * num_classes) + 1
        logits = torch.cat([negative_logits, positive_logits], dim=-1)

        # Shape: batch x seq_len x (seq_len * num_classes) + 1
        relation_scores = torch.softmax(logits, dim=-1)

        output_dict = {'logits': logits, 'relation_scores': relation_scores}

        if relations is not None:
            gold_relations = torch.zeros_like(relations)
            for example_idx in range(batch_size):
                for token_idx in range(seq_len):
                    gold_head = relation_root_idxs[example_idx, token_idx]
                    if gold_head == -1:
                        gold_relations[example_idx, token_idx] = 0
                    else:
                        gold_type = relations[example_idx, token_idx]
                        logit_idx = map_relation_head_and_type_to_index(
                            num_pos_types=self._num_pos_classes,
                            relation_head=gold_head,
                            relation_type=gold_type
                        )
                        gold_relations[example_idx, token_idx] = logit_idx

            for metric in self.metrics.values():
                metric(logits, gold_relations, mask.float())
            self._f1_metric(logits, gold_relations, mask)
            loss = sequence_cross_entropy_with_logits(logits, gold_relations, mask)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Computes the relation head and type for each word, given a tensor of
        scores for all pairwise tokens and positive relation type combinations.

        If the best score is below 0.5, the negative relation is predicted.
        """
        # pylint: disable=no-member
        relation_scores = output_dict['relation_scores']
        best_indexes = torch.max(relation_scores, dim=2)[1]

        def map_idx(idx) -> (int, str):
            return map_index_to_relation_head_and_type(self._idx_to_label, idx)

        predicted_head_offsets = []
        predicted_types = []
        for sent_best_indexes in best_indexes:
            sent_heads, sent_types = zip(*[map_idx(token_idx.item())
                                           for token_idx in sent_best_indexes])
            predicted_head_offsets.append(sent_heads)
            predicted_types.append(sent_types)

        output_dict['predicted_head_offsets'] = predicted_head_offsets
        output_dict['predicted_relations'] = predicted_types

        if 'ner_ids' in output_dict:
            predicted_heads = []
            for batch_idx, batch_ner_ids in enumerate(output_dict['ner_ids']):
                batch_predicted_heads = [
                    batch_ner_ids[offset] if offset > -1 else '-1'
                    for offset in predicted_head_offsets[batch_idx]
                ]
                predicted_heads.append(batch_predicted_heads)
            output_dict['predicted_heads'] = predicted_heads

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.metrics.items()}

        f1_dict = self._f1_metric.get_metric(reset=reset)
        if self._verbose_metrics:
            metrics_to_return.update(f1_dict)
        else:
            metrics_to_return.update({
                x: y for x, y in f1_dict.items() if
                "overall" in x})

        return metrics_to_return
