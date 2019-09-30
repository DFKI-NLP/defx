from allennlp.common.testing import ModelTestCase


class Subtask1BasicClassifierTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model('tests/fixtures/subtask1_basic_classifier.jsonnet',
                          'tests/fixtures/deft_subtask1_sample.deft')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
