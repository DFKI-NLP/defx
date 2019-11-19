from unittest import TestCase

from allennlp.data import Token
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

import defx  # Register our classes


class Subtask3PredictorTest(TestCase):
    def test_with_gold_ner(self):
        tokenized_text = [
            'Immunotherapy', 'is', 'the', 'treatment', 'of', 'disease',
            'by', 'activating', 'or', 'suppressing', 'the', 'immune',
            'system', '.',
            'Immunotherapy', 'has', 'become', 'of', 'great', 'interest', 'to',
            'researchers', '.',
            'Cell-based', 'immunotherapies', 'are', 'effective', 'for',
            'some', 'cancers', '.']
        tags = [
            'B-Term', 'O', 'B-Definition', 'I-Definition', 'I-Definition',
            'I-Definition', 'I-Definition', 'I-Definition', 'I-Definition',
            'I-Definition', 'I-Definition', 'I-Definition', 'I-Definition',
            'O', 'B-Term', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Term',
            'I-Term', 'O', 'O', 'O', 'O', 'O', 'O']
        ner_ids = [
            'T1', '-1', 'T2', 'T2', 'T2', 'T2', 'T2', 'T2', 'T2', 'T2', 'T2',
            'T2', 'T2', '-1', 'T3', '-1', '-1', '-1', '-1', '-1', '-1', '-1',
            '-1', 'T4', 'T4', '-1', '-1', '-1', '-1', '-1', '-1'
        ]
        tokens = [Token(t) for t in tokenized_text]
        json_example = {'tokens': tokens, 'tags': tags, 'ner_ids': ner_ids}

        archive = load_archive('tests/fixtures/subtask3_model_gold_ner.tar.gz')
        predictor = Predictor.from_archive(archive,
                                           'subtask3-classifier-with-gold-ner')

        result = predictor.predict_json(json_example)

        words = result.get('words')
        assert words == tokenized_text

        relations = result.get('relations')
        assert len(relations) == len(words)

        relation_head_idxs = result.get('head_offsets')
        assert len(relation_head_idxs) == len(words)

        relation_heads = result.get('heads')
        assert len(relation_heads) == len(words)
        for head in relation_heads:
            assert head in ner_ids

        logits = result.get('logits')
        assert len(logits) == len(words)
        # the model was trained on a subset of the data
        num_classes = 2
        assert len(logits[0]) == len(words) * num_classes + 1
