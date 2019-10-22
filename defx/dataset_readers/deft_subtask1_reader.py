import itertools
from pathlib import Path
from typing import Dict
import logging

from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.token import Token

logger = logging.getLogger(__name__) # pylint: disable=invalid-name


@DatasetReader.register('subtask1_reader')
class DeftSubtask1Reader(DatasetReader):
    """
    Dataset reader for the subtask 1 binary classification format.

    Expects a tsv file with two columns. The first column contains the sentence
    string with tokens separated by whitespaces. The second column contains the
    classification label, which is either ``HasDef`` or ``NoDef``

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start
        sooner, but will take longer per batch.  This also allows training
        with datasets that are too large to fit in memory.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    """

    def __init__(self,
                 lazy: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        path = Path(file_path)
        assert path.exists(), "path must be existing file or directory"

        if path.is_dir():
            return itertools.chain(*[self._read_file(f) for f in path.iterdir()])
        else:
            return self._read_file(file_path)

    def _read_file(self, file_path):
        with open(cached_path(file_path), 'r') as data_file:
            logger.info("Reading instances from %s", file_path)
            for idx, line in enumerate(data_file):
                line = line.strip('\n')
                text, label = line.split('\t')
                text = text[1:-1].strip()
                label = 'HasDef' if label == '"1"' else 'NoDef'
                origin = f'{Path(file_path).name}##{idx}'
                yield self.text_to_instance(text=text,
                                            label=label,
                                            origin=origin)

    @overrides
    def text_to_instance(self,
                         text: str,
                         label: str = None,
                         origin: str = None) -> Instance:
        # pylint: disable=arguments-differ
        tokens = [Token(t) for t in text.split(' ')]
        fields = {
            'tokens': TextField(tokens, self._token_indexers)
        }
        if label is not None:
            fields['label'] = LabelField(label)
        if origin is not None:
            fields['origin'] = MetadataField(origin)
        return Instance(fields)
