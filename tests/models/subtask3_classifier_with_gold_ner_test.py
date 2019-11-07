from allennlp.common.testing import ModelTestCase

import defx  # register classes


class Subtask3ClassifierWithGoldNerTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model('tests/fixtures/subtask3_classifier_with_gold_ner.jsonnet',
                          'tests/fixtures/jsonl_format_samples.jsonl')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
