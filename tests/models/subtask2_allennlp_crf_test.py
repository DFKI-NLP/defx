from allennlp.common.testing import ModelTestCase

import defx  # register classes


class Subtask2AllenNlpWrapperTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model('tests/fixtures/subtask2_allennlp_crf.jsonnet',
                          'tests/fixtures/jsonl_format_samples.jsonl')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
