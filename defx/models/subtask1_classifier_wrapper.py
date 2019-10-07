from typing import Dict

from overrides import overrides
import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import F1Measure


@Model.register('subtask1_classifier_wrapper')
class Subtask1ClassifierWrapper(Model):
    """
    Wraps pre-built models and uses the F1 measure instead of accuracy.

    Parameters
    ----------
    vocab : ``Vocabulary``
    model : ``Model``
        The wrapped classifier model.
    label_namespace: ``str``, optional(default = "labels")
        Vocabulary namespace corresponding to labels. By default, we use the
        "labels" namespace.
    positive_label: ``str``, optional(default = "HasDef")
        The positive class label to use for F1 metric evaluation.
    """
    
    def __init__(self,
                 vocab: Vocabulary,
                 model: Model,
                 label_namespace: str = "labels",
                 positive_label: str = "HasDef"):
        super().__init__(vocab)
        self._model = model
        label_vocab = vocab.get_token_to_index_vocabulary(label_namespace)
        self._f1_measure = F1Measure(label_vocab[positive_label])

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        output_dict = self._model(tokens, label)
        if label is not None:
            self._f1_measure(output_dict['logits'], label)
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, fscore = self._f1_measure.get_metric(reset)
        return {
            'precision': precision,
            'recall': recall,
            'f1-measure': fscore
        }
