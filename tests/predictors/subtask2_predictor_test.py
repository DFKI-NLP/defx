from unittest import TestCase

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

import defx  # Register our classes


class Subtask2PredictorTest(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
            'sentence': "5 . Science includes such diverse fields as astronomy , biology , computer sciences , geology , logic , physics , chemistry , and mathematics ( [ link ] ) ."
        }

        archive = load_archive('tests/fixtures/subtask2_model.tar.gz')
        predictor = Predictor.from_archive(archive, 'sentence-tagger')

        result = predictor.predict_json(inputs)

        words = result.get('words')
        assert " ".join(words) == inputs['sentence']

        tags = result.get('tags')
        assert len(tags) == len(words)

        logits = result.get('logits')
        assert len(logits) == len(words)
        assert len(logits[0]) == 7
