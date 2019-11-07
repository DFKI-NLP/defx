from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list
from allennlp.data import Token

from defx import DeftJsonlReader


class DeftJsonlDatasetReaderTest(AllenNlpTestCase):
    """Tests the implementation to parse deft samples in jsonl format"""

    @staticmethod
    def test_read_samples():
        """Tests parsing the samples file"""
        reader = DeftJsonlReader(subtasks=[1, 2, 3])
        instances = ensure_list(reader._read(
            'tests/fixtures/jsonl_format_samples.jsonl'))
        assert len(instances) == 5

        expected_fields = [
            "metadata", "tokens", "sentence_labels", "tags",
            "relation_root_idxs", "relations"
        ]
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
        instance_tokens = [t.text for t in instances[0].fields.get("tokens")]
        assert instance_tokens == expected_tokens

    @staticmethod
    def test_text_to_test_instance():
        """Tests the creation of a single test instance without labels"""
        reader = DeftJsonlReader(subtasks=[1, 2, 3])
        tokenized_text = [
            'Immunotherapy', 'is', 'the', 'treatment', 'of', 'disease',
            'by', 'activating', 'or', 'suppressing', 'the', 'immune',
            'system', '.',
            'Immunotherapy', 'has', 'become', 'of', 'great', 'interest', 'to',
            'researchers', '.',
            'Cell-based', 'immunotherapies', 'are', 'effective', 'for',
            'some', 'cancers', '.']
        tokens = [Token(t) for t in tokenized_text]
        instance = reader.text_to_instance(tokens)

        assert len(instance.fields) == 2
        assert 'tokens' in instance.fields
        assert 'metadata' in instance.fields

        instance_tokens = [t.text for t in instance.fields.get('tokens')]
        assert instance_tokens == tokenized_text

    @staticmethod
    def test_text_to_subtask1_instance():
        """Tests the creation of a single instance"""
        reader = DeftJsonlReader(subtasks=[1])
        example_id = 'some_example_id'
        # token spans are not correct, but that should not matter
        sentence_labels = [
            {'label': 'HasDef', 'start_token_idx': 0, 'end_token_idx': 31},
            {'label': 'NoDef', 'start_token_idx': 31, 'end_token_idx': 52},
            {'label': 'NoDef', 'start_token_idx': 52, 'end_token_idx': 68}
        ]
        tokenized_text = [
            'Immunotherapy', 'is', 'the', 'treatment', 'of', 'disease',
            'by', 'activating', 'or', 'suppressing', 'the', 'immune',
            'system', '.',
            'Immunotherapy', 'has', 'become', 'of', 'great', 'interest', 'to',
            'researchers', '.',
            'Cell-based', 'immunotherapies', 'are', 'effective', 'for',
            'some', 'cancers', '.']
        tokens = [Token(t) for t in tokenized_text]
        tags = [
            'B-Term', 'O', 'B-Definition', 'I-Definition', 'I-Definition',
            'I-Definition', 'I-Definition', 'I-Definition', 'I-Definition',
            'I-Definition', 'I-Definition', 'I-Definition', 'I-Definition',
            'O', 'B-Term', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Term',
            'I-Term', 'O', 'O', 'O', 'O', 'O', 'O']

        instance = reader.text_to_instance(
            tokens=tokens,
            example_id=example_id,
            sentence_labels=sentence_labels,
            tags=tags)

        assert len(instance.fields) == 3

        instance_tokens = [t.text for t in instance.fields.get('tokens')]
        assert instance_tokens == tokenized_text

        def get_sentence_label(idx):
            return instance.fields.get('sentence_labels')[idx].label

        assert get_sentence_label(0) == sentence_labels[0]['label']
        assert get_sentence_label(1) == sentence_labels[1]['label']
        assert get_sentence_label(2) == sentence_labels[2]['label']

        assert instance.fields.get('metadata')['sentence_offsets'] == [
            (0, 31), (31, 52), (52, 68)
        ]

        assert 'tags' not in instance.fields
        assert 'relations' not in instance.fields
        assert 'relation_root_idxs' not in instance.fields

    @staticmethod
    def test_text_to_subtask2_instance():
        """Tests the creation of a single instance"""
        reader = DeftJsonlReader(subtasks=[2])
        example_id = 'some_example_id'
        # token spans are not correct, but that should not matter
        sentence_labels = [
            {'label': 'HasDef', 'start_token_idx': 0, 'end_token_idx': 31},
            {'label': 'NoDef', 'start_token_idx': 31, 'end_token_idx': 52},
            {'label': 'NoDef', 'start_token_idx': 52, 'end_token_idx': 68}
        ]
        tokenized_text = [
            'Immunotherapy', 'is', 'the', 'treatment', 'of', 'disease',
            'by', 'activating', 'or', 'suppressing', 'the', 'immune',
            'system', '.',
            'Immunotherapy', 'has', 'become', 'of', 'great', 'interest', 'to',
            'researchers', '.',
            'Cell-based', 'immunotherapies', 'are', 'effective', 'for',
            'some', 'cancers', '.']
        tokens = [Token(t) for t in tokenized_text]
        tags = [
            'B-Term', 'O', 'B-Definition', 'I-Definition', 'I-Definition',
            'I-Definition', 'I-Definition', 'I-Definition', 'I-Definition',
            'I-Definition', 'I-Definition', 'I-Definition', 'I-Definition',
            'O', 'B-Term', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Term',
            'I-Term', 'O', 'O', 'O', 'O', 'O', 'O']

        instance = reader.text_to_instance(
            tokens=tokens,
            example_id=example_id,
            sentence_labels=sentence_labels,
            tags=tags)

        assert len(instance.fields) == 3

        # "Subtask 2 only" dataset readers should use
        # the labels namespace for the sequence tags
        assert instance.fields.get('tags')._label_namespace == 'labels'

        instance_tokens = [t.text for t in instance.fields.get('tokens')]
        assert instance_tokens == tokenized_text

        assert 'sentence_labels' not in instance.fields
        assert instance.fields.get('tags').labels == tags
        assert 'relations' not in instance.fields
        assert 'relation_root_idxs' not in instance.fields

    @staticmethod
    def test_text_to_subtask3_training_instance():
        """Tests the creation of a training instance for subtask 3 only"""
        reader = DeftJsonlReader(subtasks=[3])
        example_id = 'some_example_id'
        # token spans are not correct, but that should not matter
        sentence_labels = [
            {'label': 'HasDef', 'start_token_idx': 0, 'end_token_idx': 31},
            {'label': 'NoDef', 'start_token_idx': 31, 'end_token_idx': 52},
            {'label': 'NoDef', 'start_token_idx': 52, 'end_token_idx': 68}
        ]
        tokenized_text = [
            'Immunotherapy', 'is', 'the', 'treatment', 'of', 'disease',
            'by', 'activating', 'or', 'suppressing', 'the', 'immune',
            'system', '.',
            'Immunotherapy', 'has', 'become', 'of', 'great', 'interest', 'to',
            'researchers', '.',
            'Cell-based', 'immunotherapies', 'are', 'effective', 'for',
            'some', 'cancers', '.']
        tokens = [Token(t) for t in tokenized_text]
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
        relation_root_idxs = [
            -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        ]
        relations = [
            '0', '0', 'Direct-Defines', 'Direct-Defines', 'Direct-Defines',
            'Direct-Defines', 'Direct-Defines', 'Direct-Defines',
            'Direct-Defines', 'Direct-Defines', 'Direct-Defines',
            'Direct-Defines', 'Direct-Defines', '0', '0', '0', '0', '0', '0',
            '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'
        ]

        instance = reader.text_to_instance(
            tokens=tokens,
            example_id=example_id,
            sentence_labels=sentence_labels,
            tags=tags,
            ner_ids=ner_ids,
            relation_root_idxs=relation_root_idxs,
            relations=relations)

        assert len(instance.fields) == 5
        assert 'tokens' in instance.fields
        assert 'metadata' in instance.fields
        assert 'tags' in instance.fields
        assert 'relations' in instance.fields
        assert 'relation_root_idxs' in instance.fields

        instance_tokens = [t.text for t in instance.fields.get('tokens')]
        assert instance_tokens == tokenized_text

        assert 'sentence_labels' not in instance.fields
        assert instance.fields.get('tags').labels == tags
        assert instance.fields.get('metadata')['ner_ids'] == ner_ids
        instance_relations = instance.fields.get('relations')
        assert instance_relations.labels == relations
        assert instance_relations._label_namespace == 'relation_labels'
        instance_heads_field = instance.fields.get('relation_root_idxs')
        instance_head_idxs = [idx_field.sequence_index
                              for idx_field in instance_heads_field.field_list]
        assert instance_head_idxs == relation_root_idxs

    @staticmethod
    def test_text_to_subtask3_test_instance_with_gold_ner():
        """Tests the creation of a test instance for subtask 3 only"""
        reader = DeftJsonlReader(subtasks=[3])
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
        instance = reader.text_to_instance(tokens, tags, ner_ids)

        assert len(instance.fields) == 3
        assert 'tokens' in instance.fields
        assert 'metadata' in instance.fields
        assert 'tags' in instance.fields

        instance_text = [t.text for t in instance.fields.get('tokens')]
        assert instance_text == tokenized_text
        assert instance.fields.get('tags').labels == tags
        assert instance.fields.get('metadata')['ner_ids'] == ner_ids

        assert 'sentence_labels' not in instance.fields
        assert 'relations' not in instance.fields
        assert 'relation_root_idxs' not in instance.fields
