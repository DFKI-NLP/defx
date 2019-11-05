import itertools
import json
from typing import Dict, List, Iterable

from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, TokenIndexer, Instance, Token
from allennlp.data.fields import SequenceLabelField, MetadataField, TextField, ListField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from overrides import overrides


@DatasetReader.register('jsonl_reader')
class DeftJsonlReader(DatasetReader):
    VALID_SUBTASKS = [1, 2]

    """
    Dataset reader for deft files converted to the jsonl format.

    Expects a single jsonl file that was created by converting .deft files
    to the jsonl format using the ``DeftToJsonlConverter``.
    """
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
                sentences = example_json['sentences']

                tokens = [[Token(t) for t in s['tokens']] for s in sentences]
                if 'sentence_label' in sentences[0]:
                    sentence_labels = [s['sentence_label'] for s in sentences]
                else:
                    sentence_labels = None
                if 'tags' in sentences[0]:
                    tags = [s['tags'] for s in sentences]
                else:
                    tags = None
                yield self.text_to_instance(tokens=tokens,
                                            sentence_labels=sentence_labels,
                                            tags=tags,
                                            example_id=example_json['id'])

    @overrides
    def text_to_instance(self,
                         tokens: List[List[Token]],
                         sentence_labels: List[str] = None,
                         tags: List[List[str]] = None,
                         example_id: str = None) -> Instance:
        # pylint: disable=arguments-differ
        assert len(tokens) > 0, 'Empty example encountered'

        concatenated_tokens = list(itertools.chain.from_iterable(tokens))
        text_field = TextField(concatenated_tokens,
                               token_indexers=self._token_indexers)

        metadata = {"words": [t.text for t in concatenated_tokens]}
        if example_id:
            metadata["example_id"] = example_id
        fields = {'metadata': MetadataField(metadata), 'tokens': text_field}

        if sentence_labels and 1 in self._subtasks:
            fields['sentence_labels'] = ListField(
                [LabelField(label) for label in sentence_labels])

        if tags and 2 in self._subtasks:
            tags_field = SequenceLabelField(
                [tag for tag in itertools.chain.from_iterable(tags)],
                sequence_field=text_field,
                label_namespace=self._tags_namespace)
            fields['tags'] = tags_field

        return Instance(fields)
