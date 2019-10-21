from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('subtask1_predictor')
class Subtask1Predictor(Predictor):
    """Wrapper for subtask 1 predictors"""

    @overrides
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        # pylint: disable=arguments-differ
        instance = self._json_to_instance(json_dict)
        output_dict = self.predict_instance(instance)

        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        all_labels = [label_dict[i] for i in range(len(label_dict))]
        output_dict['all_labels'] = all_labels

        return output_dict

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(text=json_dict['text'])
