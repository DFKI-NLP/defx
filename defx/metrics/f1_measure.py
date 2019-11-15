from statistics import mean
from typing import Dict, List, Optional, Set
from collections import defaultdict

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric

from defx.util.index_to_relation_and_type_mapping import map_index_to_relation_head_and_type


@Metric.register("f1-measure")
class F1Measure(Metric):
    """
    Computes macro-averaged F1 score for relation classification on a token level.
    """
    def __init__(self,
                 vocabulary: Vocabulary,
                 negative_label: str,
                 label_namespace: str = "labels",
                 evaluated_labels: List[str] = None) -> None:

        self._label_vocabulary = vocabulary.get_index_to_token_vocabulary(label_namespace)
        if evaluated_labels is None:
            self.evaluated_labels = list(self._label_vocabulary.values())
            self.evaluated_labels.remove(negative_label)
        else:
            self.evaluated_labels = evaluated_labels

        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)
        self._support: Dict[str, int] = defaultdict(int)

        assert self._label_vocabulary[0] == negative_label, 'Negative label should have index 0'
        self._negative_label_idx = 0

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, sequence_length, num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, sequence_length).
            It must be the same shape as the ``predictions`` tensor without the
            ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        if mask is None:
            mask = torch.ones_like(gold_labels)

        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions,
                                                                gold_labels,
                                                                mask)

        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to SpanBasedF1Measure contains an "
                                     "id >= {}, the number of classes.".format(num_classes))

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        argmax_predictions = predictions.max(-1)[1]

        # Iterate over timesteps in batch.
        batch_size = gold_labels.size(0)
        for i in range(batch_size):
            sequence_prediction = argmax_predictions[i, :]
            sequence_gold_label = gold_labels[i, :]
            length = sequence_lengths[i]

            if length == 0:
                # It is possible to call this metric with sequences which are
                # completely padded. These contribute nothing, so we skip these rows.
                continue

            predicted_tuples = []
            gold_tuples = []
            for token_idx in range(length):
                pred_head_and_type_idx = sequence_prediction[token_idx].item()
                pred_head_and_type = map_index_to_relation_head_and_type(
                    label_vocab=self._label_vocabulary,
                    head_and_type_idx=pred_head_and_type_idx
                )
                pred_head, pred_label = pred_head_and_type
                if pred_label in self.evaluated_labels:
                    predicted_tuples.append(
                        (token_idx, pred_head, pred_label)
                    )

                gold_head_and_type_idx = sequence_gold_label[token_idx].item()
                gold_head_and_type = map_index_to_relation_head_and_type(
                    self._label_vocabulary,
                    gold_head_and_type_idx
                )
                gold_head, gold_label = gold_head_and_type
                if gold_label in self.evaluated_labels:
                    gold_tuples.append(
                        (token_idx, gold_head, gold_label)
                    )
                    self._support[gold_label] += 1

            for idx_head_and_type in predicted_tuples:
                relation_type = idx_head_and_type[2]
                if idx_head_and_type in gold_tuples:
                    self._true_positives[relation_type] += 1
                    gold_tuples.remove(idx_head_and_type)
                else:
                    self._false_positives[relation_type] += 1

            # These tokens weren't predicted.
            for idx_head_and_type in gold_tuples:
                self._false_negatives[idx_head_and_type[2]] += 1

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A Dict per label containing following the span based metrics:
        precision : float
        recall : float
        f1-measure : float

        Additionally, an ``overall`` key is included, which provides the precision,
        recall and f1-measure for all spans.
        """
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}

        overall_precision_values = []
        overall_recall_values = []
        overall_f1_values = []

        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(self._true_positives[tag],
                                                                  self._false_positives[tag],
                                                                  self._false_negatives[tag])
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
            support_key = "support" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure
            all_metrics[support_key] = self._support[tag]
            if tag in self.evaluated_labels:
                overall_precision_values.append(precision)
                overall_recall_values.append(recall)
                overall_f1_values.append(f1_measure)

        # If no samples are given, simply return 0
        if len(overall_precision_values) < 1:
            all_metrics["precision-overall"] = 0
            all_metrics["recall-overall"] = 0
            all_metrics["f1-measure-overall"] = 0
        else:
            all_metrics["precision-overall"] = mean(overall_precision_values)
            all_metrics["recall-overall"] = mean(overall_recall_values)
            all_metrics["f1-measure-overall"] = mean(overall_f1_values)

        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)
