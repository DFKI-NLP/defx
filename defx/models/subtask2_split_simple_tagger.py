from typing import Dict, Optional, List, Any

import numpy
import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
from overrides import overrides
from torch.nn import Linear


@Model.register('subtask2-split-simple-tagger')
class Subtask2SplitSimpleTagger(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 coarse_tag_namespace: str = 'coarse_tags',
                 modifier_tag_namespace: str = 'modifier_tags',
                 modifier_loss_weight: float = 1.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab=vocab, regularizer=regularizer)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self._modifier_loss_weight = modifier_loss_weight

        # Projections into label spaces
        encoder_output_dim = self.encoder.get_output_dim()
        self._coarse_tag_namespace = coarse_tag_namespace
        self._num_coarse_tags = self.vocab.get_vocab_size(coarse_tag_namespace)
        self._coarse_projection_layer = TimeDistributed(Linear(encoder_output_dim,
                                                               self._num_coarse_tags))
        self._modifier_tag_namespace = modifier_tag_namespace
        self._num_modifier_tags = self.vocab.get_vocab_size(modifier_tag_namespace)
        self._modifier_projection_layer = TimeDistributed(Linear(encoder_output_dim,
                                                                 self._num_modifier_tags))

        # Metrics
        self._coarse_acc = CategoricalAccuracy()
        self._modifier_acc = CategoricalAccuracy()
        self._coarse_f1 = SpanBasedF1Measure(vocab,
                                             tag_namespace=coarse_tag_namespace,
                                             label_encoding='BIO')
        self._modifier_f1 = SpanBasedF1Measure(vocab,
                                               tag_namespace=modifier_tag_namespace,
                                               label_encoding='BIO')

        initializer(self)

    @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                coarse_tags: torch.LongTensor = None,
                modifier_tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ,no-member
        embedded_text_input = self.text_field_embedder(tokens)
        batch_size, sequence_length, _ = embedded_text_input.size()
        mask = get_text_field_mask(tokens)

        # Shape: batch x seq_len x emb_dim
        encoded_text = self.encoder(embedded_text_input, mask)

        coarse_logits = self._coarse_projection_layer(encoded_text)
        reshaped_coarse_logits = coarse_logits.view(-1, self._num_coarse_tags)
        coarse_probs = F.softmax(reshaped_coarse_logits, dim=-1).view([batch_size,
                                                                       sequence_length,
                                                                       self._num_coarse_tags])

        modifier_logits = self._modifier_projection_layer(encoded_text)
        reshaped_modifier_logits = modifier_logits.view(-1, self._num_modifier_tags)
        modifier_probs = F.softmax(reshaped_modifier_logits, dim=-1).view([batch_size,
                                                                           sequence_length,
                                                                           self._num_modifier_tags])

        output_dict = {
            "coarse_logits": coarse_logits,
            "coarse_probs": coarse_probs,
            "modifier_logits": modifier_logits,
            "modifier_probs": modifier_probs,
        }

        if coarse_tags is not None or modifier_tags is not None:
            assert modifier_tags is not None, 'coarse and modifier tags should both be present'
            assert coarse_tags is not None, 'coarse and modifier tags should both be present'

            self._coarse_acc(coarse_logits, coarse_tags, mask.float())
            self._coarse_f1(coarse_logits, coarse_tags, mask.float())
            self._modifier_acc(modifier_logits, modifier_tags, mask.float())
            self._modifier_f1(modifier_logits, modifier_tags, mask.float())

            coarse_loss = sequence_cross_entropy_with_logits(coarse_logits, coarse_tags, mask)
            modifier_loss = sequence_cross_entropy_with_logits(modifier_logits, modifier_tags, mask)
            output_dict['loss'] = coarse_loss + self._modifier_loss_weight * modifier_loss

        # Attach metadata
        if metadata is not None:
            for key in metadata[0]:
                output_dict[key] = [x[key] for x in metadata]

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        coarse_probs = output_dict['coarse_probs']
        coarse_probs = coarse_probs.cpu().data.numpy()
        if coarse_probs.ndim == 3:
            coarse_pred_list = [coarse_probs[i] for i in range(coarse_probs.shape[0])]
        else:
            coarse_pred_list = [coarse_probs]
        batch_coarse_tags = []
        for predictions in coarse_pred_list:
            argmax_indices = numpy.argmax(predictions, axis=-1)
            tags = [self.vocab.get_token_from_index(x, namespace=self._coarse_tag_namespace)
                    for x in argmax_indices]
            batch_coarse_tags.append(tags)
        output_dict['coarse_tags'] = batch_coarse_tags

        modifier_probs = output_dict['modifier_probs']
        modifier_probs = modifier_probs.cpu().data.numpy()
        if modifier_probs.ndim == 3:
            modifier_pred_list = [modifier_probs[i] for i in range(modifier_probs.shape[0])]
        else:
            modifier_pred_list = [modifier_probs]
        batch_modifier_tags = []
        for predictions in modifier_pred_list:
            argmax_indices = numpy.argmax(predictions, axis=-1)
            tags = [self.vocab.get_token_from_index(x, namespace=self._modifier_tag_namespace)
                    for x in argmax_indices]
            batch_modifier_tags.append(tags)
        output_dict['modifier_tags'] = batch_modifier_tags

        batch_joined_tags = []
        for batch_idx in range(len(batch_coarse_tags)):
            joined_tags = []
            coarse_tags = batch_coarse_tags[batch_idx]
            modifier_tags = batch_modifier_tags[batch_idx]
            for coarse_tag, modifier_tag in zip(coarse_tags, modifier_tags):
                if coarse_tag == 'O':
                    joined_tags.append('O')
                elif modifier_tag == 'O':
                    joined_tags.append(coarse_tag)
                else:
                    bio_tag, coarse_class = coarse_tag.split('-', maxsplit=1)
                    _, modifier_class = modifier_tag.split('-', maxsplit=1)
                    if modifier_class == 'frag':
                        joined_tags.append(f'{bio_tag}-{coarse_class}-frag')
                    else:
                        joined_tags.append(f'{bio_tag}-{modifier_class}-{coarse_class}')
            batch_joined_tags.append(joined_tags)
        output_dict['tags'] = batch_joined_tags

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'coarse_acc': self._coarse_acc.get_metric(reset=reset),
            'coarse_f1': self._coarse_f1.get_metric(reset=reset)['f1-measure-overall'],
            'modifier_acc': self._modifier_acc.get_metric(reset=reset),
            'modifier_f1': self._modifier_f1.get_metric(reset=reset)['f1-measure-overall'],
        }

