from unittest import TestCase

from pytest import approx
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

import defx  # Register our classes


class Subtask1PredictorTest(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
            'text': " 5 . Science includes such diverse fields as astronomy , biology , computer sciences , geology , logic , physics , chemistry , and mathematics ( [ link ] ) ."
        }

        archive = load_archive('tests/fixtures/subtask1_model.tar.gz')
        predictor = Predictor.from_archive(archive, 'subtask1_predictor')

        result = predictor.predict_json(inputs)

        label = result.get('label')
        assert label in ['HasDef', 'NoDef']

        all_labels = result.get('all_labels')
        assert all_labels == ['NoDef', 'HasDef']

        class_probabilities = result.get('probs')
        assert class_probabilities is not None
        assert all(cp > 0 for cp in class_probabilities)
        assert sum(class_probabilities) == approx(1.0)
