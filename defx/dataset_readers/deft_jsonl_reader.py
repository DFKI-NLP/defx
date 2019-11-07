import json
from typing import Dict, List, Iterable

from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, TokenIndexer, Instance, Token
from allennlp.data.fields import SequenceLabelField, MetadataField, TextField, \
    ListField, LabelField, IndexField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from overrides import overrides


@DatasetReader.register('jsonl_reader')
class DeftJsonlReader(DatasetReader):
    """
    Dataset reader for deft files converted to the jsonl format.

    Expects a single jsonl file that was created by converting .deft files
    to the jsonl format using the ``DeftToJsonlConverter``.
    """
    VALID_SUBTASKS = [1, 2, 3]

    def __init__(self,
                 subtasks: List[int],
                 lazy: bool = False,
                 sample_limit: int = None,
                 token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__(lazy)
        default_indexer = {'tokens': SingleIdTokenIndexer()}
        self._token_indexers = token_indexers or default_indexer
        for subtask in subtasks:
            assert subtask in DeftJsonlReader.VALID_SUBTASKS, "invalid subtask"
        self._subtasks = subtasks
        self._sample_limit = sample_limit

        # "Subtask 2 only" dataset readers should use the labels namespace
        # for the sequence tags, but others should use 'tags'
        if self._subtasks == [2]:
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
                ner_ids = example_json['ner_ids']
                relation_roots = example_json['relation_roots']
                relation_root_idxs = [
                    ner_ids.index(root) if root in ner_ids else -1
                    for root in relation_roots
                ]
                yield self.text_to_instance(
                    tokens=[Token(t) for t in example_json['tokens']],
                    sentence_labels=example_json['sentence_labels'],
                    tags=example_json['tags'],
                    example_id=example_json['id'],
                    relations=example_json['relations'],
                    ner_ids=ner_ids,
                    relation_root_idxs=relation_root_idxs
                )

    @overrides
    def text_to_instance(self,
                         tokens: List[Token],
                         tags: List[str] = None,
                         ner_ids: List[str] = None,
                         sentence_labels: List[Dict] = None,
                         relation_root_idxs: List[int] = None,
                         relations: List[str] = None,
                         example_id: str = None) -> Instance:
        # pylint: disable=arguments-differ,too-many-arguments
        assert len(tokens) > 0, 'Empty example encountered'

        text_field = TextField(tokens, token_indexers=self._token_indexers)

        metadata = {"words": [t.text for t in tokens]}
        if example_id:
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
            tags_field = SequenceLabelField(
                labels=tags,
                sequence_field=text_field,
                label_namespace=self._tags_namespace)
            fields['tags'] = tags_field

        if relation_root_idxs and 3 in self._subtasks:
            root_idxs_field = ListField([IndexField(root_idx, text_field)
                                         for root_idx in relation_root_idxs])
            fields['relation_root_idxs'] = root_idxs_field

        if relations and 3 in self._subtasks:
            fields['relations'] = SequenceLabelField(
                labels=relations,
                sequence_field=text_field,
                label_namespace='relation_labels'
            )

        return Instance(fields)
