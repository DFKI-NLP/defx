from allennlp.common import JsonDict
from allennlp.data import Instance, DatasetReader
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides


@Predictor.register('joint-classifier')
class JointPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader):
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(
            language='en_core_web_sm',
            pos_tags=True)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        if 'tokens' in json_dict:
            tokens = json_dict['tokens']
        else:
            tokens = self._tokenizer.split_words(json_dict['sentence'])
        return self._dataset_reader.text_to_instance(tokens=tokens)

