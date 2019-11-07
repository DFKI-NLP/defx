from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor
from overrides import overrides


@Predictor.register('subtask3-classifier-with-gold-ner')
class Subtask3PredictorWithGoldNer(Predictor):

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(
            tokens=json_dict['tokens'],
            tags=json_dict['tags'],
            ner_ids=json_dict['ner_ids']
        )

