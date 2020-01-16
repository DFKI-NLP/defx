from typing import Dict, Optional, List, Any

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, TokenEmbedder, TimeDistributed
from allennlp.modules.conditional_random_field import allowed_transitions, ConditionalRandomField
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
from overrides import overrides
from torch.nn import Linear

from defx.models.relation_scorer import RelationScorer


@Model.register('joint_classifier')
class JointClassifier(Model):
    """
    Classifies NER tags and RE classes jointly. Label encoding is expected to be 'BIO'.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    ner_tag_embedder : ``Embedding``, required
        Used to embed decoded ner tags as input to the relation scorer.
    encoder : ``Seq2SeqEncoder``
        An encoder that will learn the major logic of the task.
    relation_scorer : ``RelationScorer``
        A subtask model, that performs scoring of relations between entities.
    ner_tag_namespace : ``str``
        The vocabulary namespace of ner tags.
    verbose_ner_metrics : ``bool``, optional (default=``False``)
        Prints verbose f1 score metrics per ner tag type if enabled.
    evaluated_ner_labels : ``List[str]``, optional (default=``None``)
        The list of ner tag types that are to be used for f1 score computation.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 ner_tag_embedder: TokenEmbedder,
                 encoder: Seq2SeqEncoder,
                 relation_scorer: RelationScorer,
                 ner_tag_namespace: str = 'tags',
                 verbose_ner_metrics: bool = False,
                 evaluated_ner_labels: List[str] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab=vocab, regularizer=regularizer)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self._verbose_metrics = verbose_ner_metrics

        # NER subtask 2
        self._ner_label_encoding = 'BIO'
        self._ner_tag_namespace = ner_tag_namespace
        ner_input_dim = self.encoder.get_output_dim()
        num_ner_tags = self.vocab.get_vocab_size(ner_tag_namespace)
        self.tag_projection_layer = TimeDistributed(Linear(ner_input_dim,
                                                           num_ner_tags))
        self.ner_metrics = {
            "ner_accuracy": CategoricalAccuracy(),
            "ner_accuracy3": CategoricalAccuracy(top_k=3)
        }
        if evaluated_ner_labels is None:
            ignored_classes = None
        else:
            assert self._ner_label_encoding == 'BIO', 'expected BIO encoding'
            all_ner_tags = self.vocab.get_token_to_index_vocabulary(ner_tag_namespace).keys()
            ner_tag_classes = set([bio_tag[2:]
                                   for bio_tag in all_ner_tags
                                   if len(bio_tag) > 2])
            ignored_classes = list(set(ner_tag_classes).difference(evaluated_ner_labels))
        self.ner_f1 = SpanBasedF1Measure(
            vocabulary=vocab,
            tag_namespace=ner_tag_namespace,
            label_encoding=self._ner_label_encoding,
            ignore_classes=ignored_classes
        )

        # Use constrained crf decoding with the BIO labeling scheme
        ner_labels = self.vocab.get_index_to_token_vocabulary(ner_tag_namespace)
        constraints = allowed_transitions(self._ner_label_encoding, ner_labels)

        self.crf = ConditionalRandomField(
            num_ner_tags, constraints,
            include_start_end_transitions=True
        )

        # RE subtask 3
        self.ner_tag_embedder = ner_tag_embedder
        self.relation_scorer = relation_scorer

        initializer(self)

    @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor = None,
                relation_root_idxs: torch.LongTensor = None,
                relations: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ,no-member
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        tags : torch.LongTensor
            An integer tensor containing the gold ner tag label indexes.
        relation_root_idxs : torch.LongTensor, optional (default = None)
            An integer tensor containing the gold relation head indexes for training.
        relations : torch.LongTensor, optional (default = None)
            An integer tensor containing the gold relation label indexes for training.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Additional information such as the original words and the entity ids.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_text_input = self.text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)

        # Shape: batch x seq_len x emb_dim
        encoded_text = self.encoder(embedded_text_input, mask)

        ner_logits = self.tag_projection_layer(encoded_text)
        best_ner_paths = self.crf.viterbi_tags(ner_logits, mask)

        # Just get the tags and ignore the score.
        predicted_ner_tags = []
        predicted_ner_tags_tensor = torch.zeros_like(mask)
        for ner_path, _ in best_ner_paths:
            batch_idx = len(predicted_ner_tags)
            predicted_ner_tags.append(ner_path)
            for token_idx, ner_tag_idx in enumerate(ner_path):
                predicted_ner_tags_tensor[batch_idx, token_idx] = ner_tag_idx
        # predicted_ner_tags = [x for x, y in best_ner_paths]

        output_dict = {
            "ner_logits": ner_logits,
            "mask": mask,
            "tags": predicted_ner_tags
        }

        embedded_tags = self.ner_tag_embedder(predicted_ner_tags_tensor)
        encoded_sequence = torch.cat([encoded_text, embedded_tags], dim=2)

        re_output = self.relation_scorer(encoded_sequence,
                                         mask,
                                         relation_root_idxs,
                                         relations)

        # Add a prefix for relation extraction logits
        output_dict['re_logits'] = re_output['logits']
        output_dict['relation_scores'] = re_output['relation_scores']

        if tags is not None:
            # Add negative log-likelihood as loss
            log_likelihood = self.crf(ner_logits, tags, mask)

            # It's not clear why, but pylint seems to think `log_likelihood` is tuple
            # (in fact, it's a torch.Tensor), so we need a disable.
            output_dict["ner_loss"] = -log_likelihood  # pylint: disable=invalid-unary-operand-type

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = torch.zeros_like(ner_logits)
            for i, instance_tags in enumerate(predicted_ner_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            for metric in self.ner_metrics.values():
                metric(class_probabilities, tags, mask.float())
            self.ner_f1(class_probabilities, tags, mask.float())

            output_dict['loss'] = output_dict['ner_loss'] + re_output['loss']

        # Attach metadata
        if metadata is not None:
            for key in metadata[0]:
                output_dict[key] = [x[key] for x in metadata]

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output_dict = self.relation_scorer.decode(output_dict)
        # for key in ['relations', 'heads', 'head_offsets']:
        #     if key in re_output_dict:
        #         output_dict[key] = re_output_dict[key]
        output_dict["tags"] = [
            [self.vocab.get_token_from_index(tag, self._ner_tag_namespace)
             for tag in instance_tags]
            for instance_tags in output_dict["tags"]
        ]
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # Set NER metrics
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.ner_metrics.items()}
        ner_metrics = self.ner_f1.get_metric(reset=reset)
        if self._verbose_metrics:
            metrics_to_return.update({
                'ner_' + x: y for x, y in ner_metrics.items()
            })
        else:
            metrics_to_return['ner_f1'] = ner_metrics['f1-measure-overall']

        # Attach RE metrics
        re_metrics = self.relation_scorer.get_metrics(reset=reset)
        metrics_to_return['re_f1'] = re_metrics['f1-measure-overall']

        return metrics_to_return

