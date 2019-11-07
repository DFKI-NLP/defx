from unittest import TestCase
from pathlib import Path

from defx.util import deft_to_jsonl_converter


class DeftToJsonlConverterTest(TestCase):
    """Tests the conversion script from deft format into jsonl"""

    @staticmethod
    def test_chapter_detection():
        true_example_1 = {'tokens': ['5', '.']}
        assert deft_to_jsonl_converter._is_chapter_start(true_example_1)

        true_example_2 = {'tokens': ['5320032', '.']}
        assert deft_to_jsonl_converter._is_chapter_start(true_example_2)

        false_example_1 = {'tokens': ['5d', '.']}
        assert not deft_to_jsonl_converter._is_chapter_start(false_example_1)

        false_example_2 = {'tokens': ['.']}
        assert not deft_to_jsonl_converter._is_chapter_start(false_example_2)

        false_example_3 = {'tokens': ['Some', 'regular', 'sentence', '.']}
        assert not deft_to_jsonl_converter._is_chapter_start(false_example_3)

    @staticmethod
    def test_parse_sentence_with_chapter():
        with open('tests/fixtures/deft_format_sentence_with_chapter.deft') as f:
            example = deft_to_jsonl_converter._parse_example(f)
            assert example['tokens'] == [
                "5", ".", "Science", "includes", "such", "diverse", "fields",
                "as", "astronomy", ",", "biology", ",", "computer", "sciences",
                ",", "geology", ",", "logic", ",", "physics", ",", "chemistry",
                ",", "and", "mathematics", "(", "[", "link", "]", ")", "."
            ]
            assert example['tags'] == ['O'] * 31
            assert len(example['ner_ids']) == 31
            assert example['ner_ids'] == ['-1'] * 31
            assert example['sentence_labels'] == [{
                'label': 'NoDef',
                'start_token_idx': 0,
                'end_token_idx': 31
            }]

    @staticmethod
    def test_parse_sentence_with_annotations():
        with open('tests/fixtures/deft_format_sentence_with_annotations.deft') as f:
            example = deft_to_jsonl_converter._parse_example(f)
        assert example['tokens'][:4] == ["However", ",", "those", "fields"]
        assert example['tokens'][-3:] == ["natural", "sciences", "."]
        assert len(example['tokens']) == 21
        assert example['sentence_labels'] == [{
            'label': 'HasDef',
            'start_token_idx': 0,
            'end_token_idx': 21
        }]

        assert example['tags'][2] == "B-Definition"
        assert example['tags'][3:16] == ["I-Definition"] * 13
        assert example['ner_ids'][2:16] == ["T127"] * 14

        assert example['tags'][18] == "B-Term"
        assert example['tags'][19] == "I-Term"
        assert example['ner_ids'][18:20] == ["T128"] * 2

        assert example['relation_roots'][2:16] == ['T128'] * 14
        assert example['relations'][2:16] == ['Direct-Defines'] * 14

        # Relations pointing to the token itself should be ignored
        assert example['relation_roots'][18:20] == ['-1'] * 2
        assert example['relations'][18:20] == ['0'] * 2

    @staticmethod
    def test_convert_file():
        input_file = Path('tests/fixtures/deft_format_samples.deft')
        examples = deft_to_jsonl_converter._convert_deft_file(input_file)
        assert len(examples) == 2

        assert examples[0]['tokens'] == [
            "5", ".", "Science", "includes", "such", "diverse", "fields", "as",
            "astronomy", ",", "biology", ",", "computer", "sciences", ",",
            "geology", ",", "logic", ",", "physics", ",", "chemistry", ",",
            "and", "mathematics", "(", "[", "link", "]", ")", ".", "However",
            ",", "those", "fields", "of", "science", "related", "to", "the",
            "physical", "world", "and", "its", "phenomena", "and", "processes",
            "are", "considered", "natural", "sciences", ".", "Thus", ",", "a",
            "museum", "of", "natural", "sciences", "might", "contain", "any",
            "of", "the", "items", "listed", "above", "."
        ]

        assert examples[0]['sentence_labels'] == [
            {
                'label': 'NoDef',
                'start_token_idx': 0,
                'end_token_idx': 31
            },
            {
                'label': 'HasDef',
                'start_token_idx': 31,
                'end_token_idx': 52
            },
            {
                'label': 'NoDef',
                'start_token_idx': 52,
                'end_token_idx': 68
            }
        ]
