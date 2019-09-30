from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.data import Vocabulary
from allennlp.models import BasicClassifier
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import F1Measure


@Model.register('subtask1_basic_classifier')
class Subtask1BasicClassifier(BasicClassifier):
    # pylint: disable=too-many-arguments
    """
    This ``Model`` extends the allennlp basic text classifier. F1 measure is
    added as a metric.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the input text into a ``TextField``
    seq2seq_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        Optional Seq2Seq encoder layer for the input text.
    seq2vec_encoder : ``Seq2VecEncoder``
        Required Seq2Vec encoder layer. If `seq2seq_encoder` is provided, this
        encoder will pool its output. Otherwise, this encoder will operate
        directly on the output of the `text_field_embedder`.
    dropout : ``float``, optional (default = ``None``)
        Dropout percentage to use.
    num_labels: ``int``, optional (default = ``None``)
        Number of labels to project to in classification layer. By default, the
        classification layer will project to the size of the vocabulary
        namespace corresponding to labels.
    label_namespace: ``str``, optional (default = "labels")
        Vocabulary namespace corresponding to labels. By default, we use the
        "labels" namespace.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        If provided, will be used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty
        during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 seq2vec_encoder: Seq2VecEncoder,
                 seq2seq_encoder: Seq2SeqEncoder = None,
                 dropout: float = None,
                 num_labels: int = None,
                 label_namespace: str = "labels",
                 positive_label: str = "HasDef",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, text_field_embedder, seq2vec_encoder,
                         seq2seq_encoder, dropout, num_labels, label_namespace,
                         initializer, regularizer)
        label_vocab = vocab.get_token_to_index_vocabulary(label_namespace)
        self._f1_measure = F1Measure(label_vocab[positive_label])

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=no-member
        output_dict = super().forward(tokens, label)
        if label is not None:
            self._f1_measure(output_dict['logits'], label)
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = super().get_metrics(reset)
        precision, recall, fscore = self._f1_measure.get_metric(reset)
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1-measure'] = fscore
        return metrics
