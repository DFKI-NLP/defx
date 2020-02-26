import json
import math
from random import random
from typing import Dict, List, Iterable, Any, Tuple

import numpy as np
from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, TokenIndexer, Instance, Token
from allennlp.data.fields import SequenceLabelField, MetadataField, TextField, \
    ListField, LabelField, IndexField, AdjacencyField, ArrayField, SequenceField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from overrides import overrides

from defx import EVALUATED_SUBTASK2_LABELS, EVALUATED_SUBTASK3_LABELS


@DatasetReader.register('jsonl_reader')
class DeftJsonlReader(DatasetReader):
    """
    Dataset reader for deft files converted to the jsonl format.

    Expects a single jsonl file that was created by converting .deft files
    to the jsonl format using the ``DeftToJsonlConverter``.
    """
    VALID_SUBTASKS = [1, 2, 3]
    VALID_MAJORITY_DESCRIPTIONS = ['term_def_only', 'term_def_pair']

    def __init__(self,
                 subtasks: List[int],
                 lazy: bool = False,
                 sample_limit: int = None,
                 split_ner_labels: bool = False,
                 aux_ner_labels: bool = False,
                 aux_re_task: bool = False,
                 oversampling_ratio: float = None,
                 majority_description: str = None,
                 read_spacy_pos_tags: bool = True,
                 read_spacy_dep_rels: bool = True,
                 read_spacy_dep_heads: bool = False,
                 read_binary_coref: bool = False,
                 read_embedded_coref: bool = False,
                 add_dep_self_loops: bool = True,
                 only_evaluated_subtask2_labels: bool = False,
                 only_evaluated_subtask3_labels: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__(lazy)
        for subtask in subtasks:
            assert subtask in DeftJsonlReader.VALID_SUBTASKS, "invalid subtask"
        if majority_description is not None:
            assert majority_description in DeftJsonlReader.VALID_MAJORITY_DESCRIPTIONS

        default_indexer = {'tokens': SingleIdTokenIndexer()}
        self._token_indexers = token_indexers or default_indexer
        self._subtasks = subtasks
        self._sample_limit = sample_limit
        self._split_ner_labels = split_ner_labels
        self._aux_ner_labels = aux_ner_labels
        self._aux_re_task = aux_re_task
        self._oversampling_ratio = oversampling_ratio
        self._majority_description = majority_description
        self._read_spacy_pos_tags = read_spacy_pos_tags
        self._read_spacy_dep_rels = read_spacy_dep_rels
        self._read_spacy_dep_heads = read_spacy_dep_heads
        self._add_dep_self_loops = add_dep_self_loops
        self._only_evaluated_subtask2_labels = only_evaluated_subtask2_labels
        self._only_evaluated_subtask3_labels = only_evaluated_subtask3_labels
        self._read_binary_coref = read_binary_coref
        self._read_embedded_coref = read_embedded_coref
        if self._read_embedded_coref:
            raise NotImplementedError('coref cluster embedding not impplemented yet')

        # "Subtask 2 only" dataset readers should use the labels namespace
        # for the sequence tags, but others should use 'tags'
        if self._subtasks == [2] and not split_ner_labels:
            self._tags_namespace = 'labels'
        else:
            self._tags_namespace = 'tags'

    @overrides
    def _read(self, file_path) -> Iterable[Instance]:
        with open(cached_path(file_path), 'r') as data_file:
            for idx, line in enumerate(data_file):
                if self._sample_limit and idx >= self._sample_limit:
                    break
                example_json = json.loads(line.strip())

                tokens = []
                text_tokens = example_json['tokens']
                for token_idx, text_token in enumerate(text_tokens):
                    if self._read_spacy_pos_tags:
                        spacy_pos = example_json['spacy_pos'][token_idx]
                        spacy_tag = example_json['spacy_tag'][token_idx]
                    else:
                        spacy_pos = None
                        spacy_tag = None
                    if self._read_spacy_dep_rels:
                        spacy_dep = example_json['spacy_dep_rel'][token_idx]
                    else:
                        spacy_dep = None
                    token = Token(text=text_token, pos_=spacy_pos, tag_=spacy_tag, dep_=spacy_dep)
                    tokens.append(token)

                dep_heads = example_json['spacy_dep_head'] if 'spacy_dep_head' in example_json else None

                # Meta information
                example_id = example_json['id'] if 'id' in example_json else None

                # Subtask 1 labels
                sentence_labels = example_json['sentence_labels'] if 'sentence_labels' in example_json else None

                # Subtask 2 labels
                tags = example_json['tags'] if 'tags' in example_json else None

                # Subtask 3 labels
                ner_ids = example_json['ner_ids'] if 'ner_ids' in example_json else None
                relations = example_json['relations'] if 'relations' in example_json else None
                if 'relation_roots' in example_json:
                    assert ner_ids is not None, 'NER ids are required for RE as an auxiliary task'
                    entity_ids = set(ner_ids)
                    entity_ids.remove('-1')
                    relation_root_idxs = [
                        ner_ids.index(root) if root in entity_ids else -1
                        for root in example_json['relation_roots']
                    ]

                    if self._aux_re_task:
                        aux_relations = [['0' for _ in range(len(tokens))]
                                         for _ in range(len(tokens))]
                        entity_spans = {}
                        for ner_id in entity_ids:
                            entity_idxs = [idx
                                           for idx, token_ner_id in enumerate(ner_ids)
                                           if token_ner_id == ner_id]
                            entity_spans[ner_id] = (entity_idxs[0], entity_idxs[-1] + 1)
                        for token_idx, root_id in enumerate(example_json['relation_roots']):
                            if root_id == '-1':
                                continue
                            relation_label = relations[token_idx]
                            root_start, root_end = entity_spans[root_id]
                            aux_relations[token_idx][root_start:root_end] = [relation_label] * (root_end - root_start)
                    else:
                        aux_relations = None
                else:
                    relation_root_idxs = None
                    aux_relations = None

                if self._read_binary_coref or self._read_embedded_coref:
                    assert 'coref_clusters' in example_json, 'coref annotations are missing'
                    is_in_coref = [self._is_in_coref_cluster(idx, example_json['coref_clusters'])
                                    for idx in range(len(tokens))]
                    if self._read_binary_coref:
                        binary_coref = [float(val) for val in is_in_coref]
                    else:
                        binary_coref = None
                    if self._read_embedded_coref:
                        embedded_coref = [int(val) for val in is_in_coref]
                    else:
                        embedded_coref = [int(val) for val in is_in_coref]
                else:
                    binary_coref = None
                    embedded_coref = None

                instance = self.text_to_instance(
                    tokens=tokens,
                    sentence_labels=sentence_labels,
                    tags=tags,
                    example_id=example_id,
                    relations=relations,
                    ner_ids=ner_ids,
                    relation_root_idxs=relation_root_idxs,
                    aux_relations=aux_relations,
                    dep_heads=dep_heads,
                    binary_coref=binary_coref,
                    embedded_coref=embedded_coref,
                )
                if file_path.__str__().__contains__('train.jsonl') and self._oversampling_ratio is not None:
                    if self._is_majority_example(example_json):
                        yield instance
                    else:
                        for _ in range(self._get_num_samples()):
                            yield instance
                else:
                    yield instance

    @overrides
    def text_to_instance(self,
                         tokens: List[Token],
                         tags: List[str] = None,
                         ner_ids: List[str] = None,
                         sentence_labels: List[Dict] = None,
                         relation_root_idxs: List[int] = None,
                         aux_relations: List[List[str]] = None,
                         relations: List[str] = None,
                         example_id: str = None,
                         dep_heads: List[int] = None,
                         binary_coref: List[float] = None,
                         embedded_coref: List[int] = None,) -> Instance:
        # pylint: disable=arguments-differ,too-many-arguments
        assert len(tokens) > 0, 'Empty example encountered'

        text_field = TextField(tokens, token_indexers=self._token_indexers)

        metadata = {"words": [t.text for t in tokens]}
        if example_id is not None:
            metadata["example_id"] = example_id
        if ner_ids and 3 in self._subtasks:
            metadata['ner_ids'] = ner_ids
        if sentence_labels:
            metadata['sentence_offsets'] = [
                (label_dict['start_token_idx'], label_dict['end_token_idx'])
                for label_dict in sentence_labels
            ]

        fields = {'metadata': MetadataField(metadata), 'tokens': text_field}

        if sentence_labels and 1 in self._subtasks:
            fields['sentence_labels'] = ListField(
                [LabelField(label_dict['label'])
                 for label_dict in sentence_labels])

        if tags and (2 in self._subtasks or self._subtasks == [3]):
            if self._only_evaluated_subtask2_labels:
                tags = [tag if len(tag) > 2 and tag[2:] in EVALUATED_SUBTASK2_LABELS else 'O'
                        for tag in tags]

            if self._split_ner_labels or self._aux_ner_labels:
                assert not (self._split_ner_labels and self._aux_ner_labels), 'Can not be used together'
                coarse_tags, modifier_tags = self._split_tags(tags, use_bio=self._split_ner_labels)
                coarse_tags_field = SequenceLabelField(
                    labels=coarse_tags,
                    sequence_field=text_field,
                    label_namespace='coarse_tags')
                fields['coarse_tags'] = coarse_tags_field
                modifier_tags_field = SequenceLabelField(
                    labels=modifier_tags,
                    sequence_field=text_field,
                    label_namespace='modifier_tags')
                fields['modifier_tags'] = modifier_tags_field

            if not self._split_ner_labels:
                tags_field = SequenceLabelField(
                    labels=tags,
                    sequence_field=text_field,
                    label_namespace=self._tags_namespace)
                fields['tags'] = tags_field

        if relation_root_idxs and 3 in self._subtasks and not self._aux_re_task:
            if self._only_evaluated_subtask3_labels:
                relation_root_idxs = [
                    root_idx if relations[idx] in EVALUATED_SUBTASK3_LABELS else 0
                    for idx, root_idx in enumerate(relation_root_idxs)
                ]
            root_idxs_field = ListField([IndexField(root_idx, text_field)
                                         for root_idx in relation_root_idxs])
            fields['relation_root_idxs'] = root_idxs_field

        if relations and 3 in self._subtasks and not self._aux_re_task:
            if self._only_evaluated_subtask3_labels:
                relations = [
                    relation if relation in EVALUATED_SUBTASK3_LABELS else '0'
                    for relation in relations
                ]
            fields['relations'] = SequenceLabelField(
                labels=relations,
                sequence_field=text_field,
                label_namespace='relation_labels'
            )

        if aux_relations and 3 in self._subtasks and self._aux_re_task:
            assert not self._only_evaluated_subtask3_labels, 'Only eval labels not supported for auxialiary task re'
            aux_relations_list_field = ListField([
                SequenceLabelField(labels=aux_token_relations,
                                   sequence_field=text_field,
                                   label_namespace='relation_labels')
                for aux_token_relations in aux_relations
            ])
            fields['relations'] = aux_relations_list_field

        if self._read_spacy_dep_heads:
            assert dep_heads is not None, 'Dependency head indexes are missing'
            indices = self._parse_adjacency_indices(dep_heads, add_self_loops=self._add_dep_self_loops)
            fields["adjacency"] = AdjacencyField(indices, sequence_field=text_field, padding_value=0)

        if self._read_binary_coref:
            assert binary_coref, 'Binary coref missing'
            fields["binary_coref"] = ArrayField(np.array(binary_coref), padding_value=-1.0)

        if self._read_embedded_coref:
            assert embedded_coref, 'Embedding value for coref missing'
            fields["embedded_coref"] = ArrayField(np.array(embedded_coref), padding_value=-1)

        return Instance(fields)

    def _split_tags(self, tags: List[str], use_bio=True) -> (List[str], List[str]):
        coarse_tags = []
        modifier_tags = []
        for tag in tags:
            if tag == 'O':
                coarse_tags.append('O')
                modifier_tags.append('O')
            else:
                tag_splits = tag.split('-')

                if len(tag_splits) == 2:
                    bio_tag, coarse_tag = tag_splits
                    modifier_tag = None
                elif len(tag_splits) == 3:
                    bio_tag, modifier_tag, coarse_tag = tag_splits
                    if coarse_tag == 'frag':
                        coarse_tag = modifier_tag
                        modifier_tag = 'frag'
                elif len(tag_splits) == 4:
                    assert tag_splits[-1] == 'frag'
                    # Just ignore this very special case, e.g. 'B-Alias-Term-frag'
                    coarse_tags.append('O')
                    modifier_tags.append('O')
                    continue
                else:
                    raise RuntimeError(f'Unexpected ner tag encountered: {tag}')

                assert coarse_tag in ['Term', 'Definition', 'Qualifier'], f'Unknown coarse tag: {coarse_tag}'
                if use_bio:
                    coarse_tags.append(bio_tag + '-' + coarse_tag)
                else:
                    coarse_tags.append(coarse_tag)

                if modifier_tag is None:
                    modifier_tags.append('O')
                else:
                    assert modifier_tag in ['Alias', 'Ordered', 'Referential', 'Secondary', 'frag'], f'Unknown modifier tag: {modifier_tag}'
                    if use_bio:
                        modifier_tags.append(bio_tag + '-' + modifier_tag)
                    else:
                        modifier_tags.append(modifier_tag)
        return coarse_tags, modifier_tags

    def _get_num_samples(self) -> int:
        num_deterministic_samples = math.floor(self._oversampling_ratio)
        additional_sample_prob = self._oversampling_ratio - num_deterministic_samples
        if additional_sample_prob > random():
            num_samples = num_deterministic_samples + 1
        else:
            num_samples = num_deterministic_samples
        return num_samples

    @staticmethod
    def _is_in_coref_cluster(idx, clusters):
        for cluster in clusters:
            for span in cluster:
                if span[0] <= idx <= span[1]:
                    return True
        return False

    def _is_majority_example(self, example: Dict[str, Any]) -> bool:
        tags = example['tags']
        if self._majority_description == 'term_def_only':
            unique_tags = set([tag.split('-', maxsplit=1)[1] for tag in tags if tag != 'O'])
            return len(unique_tags) == 0 or unique_tags == {'Definition', 'Term'}
        elif self._majority_description == 'term_def_pair':
            sorted_start_tags = sorted([tag for tag in tags if tag.startswith('B-')])
            return len(sorted_start_tags) == 0 or sorted_start_tags == ['B-Definition', 'B-Term']
        else:
            raise RuntimeError('unknown description for the majority of the examples')

    @staticmethod
    def _parse_adjacency_indices(head_idxs: List[int],
                                 add_self_loops: bool = False) -> List[Tuple[int, int]]:
        indices = [(node_idx, head_idx)
                   for node_idx, head_idx in enumerate(head_idxs)
                   if node_idx != head_idx]
        if add_self_loops:
            indices.extend([(node_idx, node_idx)
                            for node_idx in range(len(head_idxs))])
        return indices

