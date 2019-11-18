from typing import Dict, Optional, List, Any

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, TokenEmbedder
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from overrides import overrides

from defx.models.relation_scorer import RelationScorer


@Model.register('subtask3_classifier_with_gold_ner')
class Subtask3ClassifierWithGoldNer(Model):
    """
    Classifies relations for subtask 3 using gold ner tags.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    ner_tag_embedder : ``Embedding``, required
        Used to embed the gold ner tags we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        An encoder that will learn the major logic of the task.
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
                 ignore_ner: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab=vocab, regularizer=regularizer)

        self.text_field_embedder = text_field_embedder
        self.ner_tag_embedder = ner_tag_embedder
        self.relation_scorer = relation_scorer
        self.encoder = encoder
        self._ner_tag_namespace = ner_tag_namespace
        self._ignore_ner = ignore_ner

        initializer(self)

    @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor,
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
        if self._ignore_ner:
            encoded_sequence = encoded_text
        else:
            embedded_tags = self.ner_tag_embedder(tags)
            encoded_sequence = torch.cat([encoded_text, embedded_tags], dim=2)

        output_dict = self.relation_scorer(encoded_sequence,
                                           mask,
                                           relation_root_idxs,
                                           relations)

        # Attach metadata
        if metadata is not None:
            for key in metadata[0]:
                output_dict[key] = [x[key] for x in metadata]

        # Store the ner tags in the output for later decoding
        output_dict["tags"] = tags

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output_dict = self.relation_scorer.decode(output_dict)
        output_dict["tags"] = [
            [self.vocab.get_token_from_index(tag.item(), namespace=self._ner_tag_namespace)
             for tag in instance_tags]
            for instance_tags in output_dict["tags"]
        ]
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.relation_scorer.get_metrics(reset=reset)
