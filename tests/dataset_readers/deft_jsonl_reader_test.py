import itertools

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from defx import DeftJsonlReader


class DeftJsonlDatasetReaderTest(AllenNlpTestCase):
    """Tests the implementation to parse deft samples in jsonl format"""

    @staticmethod
    def test_read_samples():
        """Tests parsing the samples file"""
        reader = DeftJsonlReader(subtasks=[1, 2])
        instances = ensure_list(reader._read(
            'tests/fixtures/jsonl_format_samples.jsonl'))
        assert len(instances) == 5

        expected_fields = ["metadata", "tokens", "sentence_labels", "tags"]
        for instance in instances:
            assert list(instance.fields.keys()) == expected_fields

        expected_tokens = [
            "3616", ".", "Some", "of", "these", "are", "binocular", "cues",
            ",", "which", "means", "that", "they", "rely", "on", "the", "use",
            "of", "both", "eyes", ".", "One", "example", "of", "a",
            "binocular", "depth", "cue", "is", "binocular", "disparity", ",",
            "the", "slightly", "different", "view", "of", "the", "world",
            "that", "each", "of", "our", "eyes", "receives", ".", "To",
            "experience", "this", "slightly", "different", "view", ",", "do",
            "this", "simple", "exercise", ":", "extend", "your", "arm", "fully",
            "and", "extend", "one", "of", "your", "fingers", "and", "focus",
            "on", "that", "finger", "."
        ]
        metadata_field = instances[0].fields.get('metadata')
        assert metadata_field['words'] == expected_tokens
        tokens = [t.text for t in instances[0].fields.get("tokens")]
        assert tokens == expected_tokens

    @staticmethod
    def test_text_to_instance():
        """Tests the creation of a single instance"""
        reader = DeftJsonlReader(subtasks=[1, 2])
        example_id = 'some_example_id'
        sentence_labels = ['HasDef', 'NoDef', 'NoDef']
        tokens = [
            ['Immunotherapy', 'is', 'the', 'treatment', 'of', 'disease',
             'by', 'activating', 'or', 'suppressing', 'the', 'immune',
             'system', '.'],
            ['Immunotherapy', 'has', 'become', 'of', 'great', 'interest', 'to',
             'researchers', '.'],
            ['Cell-based', 'immunotherapies', 'are', 'effective', 'for',
             'some', 'cancers', '.']]
        tags = [
            ['B-Term', 'O', 'B-Definition', 'I-Definition', 'I-Definition',
             'I-Definition', 'I-Definition', 'I-Definition', 'I-Definition',
             'I-Definition', 'I-Definition', 'I-Definition', 'I-Definition',
             'O'],
            ['B-Term', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['B-Term', 'I-Term', 'O', 'O', 'O', 'O', 'O', 'O']]

        instance = reader.text_to_instance(
            tokens=tokens,
            example_id=example_id,
            sentence_labels=sentence_labels,
            tags=tags)

        assert len(instance.fields) == 4

        expected_tokens = list(itertools.chain.from_iterable(tokens))

        metadata_field = instance.fields.get('metadata')
        assert metadata_field['words'] == expected_tokens
        assert metadata_field['example_id'] == example_id

        tokens = [t.text for t in instance.fields.get('tokens')]
        assert tokens == expected_tokens

        def get_sentence_label(idx):
            return instance.fields.get('sentence_labels')[idx].label
        assert get_sentence_label(0) == sentence_labels[0]
        assert get_sentence_label(1) == sentence_labels[1]
        assert get_sentence_label(2) == sentence_labels[2]

        expected_tags = itertools.chain.from_iterable(tags)
        assert instance.fields.get('tags').labels == list(expected_tags)

    @staticmethod
    def test_text_to_test_instance():
        """Tests the creation of a single test instance without labels"""
        reader = DeftJsonlReader(subtasks=[1, 2])
        tokens = [
            ['Immunotherapy', 'is', 'the', 'treatment', 'of', 'disease',
             'by', 'activating', 'or', 'suppressing', 'the', 'immune',
             'system', '.'],
            ['Immunotherapy', 'has', 'become', 'of', 'great', 'interest', 'to',
             'researchers', '.'],
            ['Cell-based', 'immunotherapies', 'are', 'effective', 'for',
             'some', 'cancers', '.']]

        instance = reader.text_to_instance(tokens)

        assert len(instance.fields) == 2

        expected_tokens = itertools.chain.from_iterable(tokens)
        tokens = [t.text for t in instance.fields.get('tokens')]
        assert tokens == list(expected_tokens)

    @staticmethod
    def test_text_to_subtask1_instance():
        """Tests the creation of a single instance"""
        reader = DeftJsonlReader(subtasks=[1])
        example_id = 'some_example_id'
        sentence_labels = ['HasDef', 'NoDef', 'NoDef']
        tokens = [
            ['Immunotherapy', 'is', 'the', 'treatment', 'of', 'disease',
             'by', 'activating', 'or', 'suppressing', 'the', 'immune',
             'system', '.'],
            ['Immunotherapy', 'has', 'become', 'of', 'great', 'interest', 'to',
             'researchers', '.'],
            ['Cell-based', 'immunotherapies', 'are', 'effective', 'for',
             'some', 'cancers', '.']]
        tags = [
            ['B-Term', 'O', 'B-Definition', 'I-Definition', 'I-Definition',
             'I-Definition', 'I-Definition', 'I-Definition', 'I-Definition',
             'I-Definition', 'I-Definition', 'I-Definition', 'I-Definition',
             'O'],
            ['B-Term', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['B-Term', 'I-Term', 'O', 'O', 'O', 'O', 'O', 'O']]

        instance = reader.text_to_instance(
            tokens=tokens,
            example_id=example_id,
            sentence_labels=sentence_labels,
            tags=tags)

        assert len(instance.fields) == 3

        expected_tokens = itertools.chain.from_iterable(tokens)
        tokens = [t.text for t in instance.fields.get('tokens')]
        assert tokens == list(expected_tokens)

        def get_sentence_label(idx):
            return instance.fields.get('sentence_labels')[idx].label
        assert get_sentence_label(0) == sentence_labels[0]
        assert get_sentence_label(1) == sentence_labels[1]
        assert get_sentence_label(2) == sentence_labels[2]

        assert 'tags' not in instance.fields

    @staticmethod
    def test_text_to_subtask2_instance():
        """Tests the creation of a single instance"""
        reader = DeftJsonlReader(subtasks=[2])
        example_id = 'some_example_id'
        sentence_labels = ['HasDef', 'NoDef', 'NoDef']
        tokens = [
            ['Immunotherapy', 'is', 'the', 'treatment', 'of', 'disease',
             'by', 'activating', 'or', 'suppressing', 'the', 'immune',
             'system', '.'],
            ['Immunotherapy', 'has', 'become', 'of', 'great', 'interest', 'to',
             'researchers', '.'],
            ['Cell-based', 'immunotherapies', 'are', 'effective', 'for',
             'some', 'cancers', '.']]
        tags = [
            ['B-Term', 'O', 'B-Definition', 'I-Definition', 'I-Definition',
             'I-Definition', 'I-Definition', 'I-Definition', 'I-Definition',
             'I-Definition', 'I-Definition', 'I-Definition', 'I-Definition',
             'O'],
            ['B-Term', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['B-Term', 'I-Term', 'O', 'O', 'O', 'O', 'O', 'O']]

        instance = reader.text_to_instance(
            tokens=tokens,
            example_id=example_id,
            sentence_labels=sentence_labels,
            tags=tags)

        assert len(instance.fields) == 3

        expected_tokens = itertools.chain.from_iterable(tokens)
        tokens = [t.text for t in instance.fields.get('tokens')]
        assert tokens == list(expected_tokens)

        assert 'sentence_labels' not in instance.fields

        expected_tags = itertools.chain.from_iterable(tags)
        assert instance.fields.get('tags').labels == list(expected_tags)
