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
            sentence = deft_to_jsonl_converter._parse_sentence(f)
            assert sentence['tokens'] == [
                "5", ".", "Science", "includes", "such", "diverse", "fields",
                "as", "astronomy", ",", "biology", ",", "computer", "sciences",
                ",", "geology", ",", "logic", ",", "physics", ",", "chemistry",
                ",", "and", "mathematics", "(", "[", "link", "]", ")", "."
            ]

            assert sentence['tags'] == ['O'] * 31
            assert len(sentence['ner_ids']) == 31
            assert sentence['ner_ids'] == ['-1'] * 31
            assert sentence['sentence_label'] == 'NoDef'

    @staticmethod
    def test_parse_sentence_with_annotations():
        with open('tests/fixtures/deft_format_sentence_with_annotations.deft') as f:
            sentence = deft_to_jsonl_converter._parse_sentence(f)
        assert sentence['tokens'][:4] == ["However", ",", "those", "fields"]
        assert sentence['tokens'][-3:] == ["natural", "sciences", "."]
        assert len(sentence['tokens']) == 21
        assert sentence['sentence_label'] == 'HasDef'

        entities = deft_to_jsonl_converter._extract_entities([sentence])
        assert len(entities) == 2

        assert entities[0]['id'] == "T127"
        assert entities[0]['entity_type'] == "Definition"
        assert entities[0]['start_token'] == 2
        assert entities[0]['end_token'] == 16

        assert entities[1]['id'] == "T128"
        assert entities[1]['entity_type'] == "Term"
        assert entities[1]['start_token'] == 18
        assert entities[1]['end_token'] == 20

        relations = deft_to_jsonl_converter._extract_relations([sentence])
        assert len(relations) == 1

        assert relations[0]['head_id'] == 'T128'
        assert relations[0]['tail_id'] == 'T127'
        assert relations[0]['relation_type'] == 'Direct-Defines'

    @staticmethod
    def test_convert_file():
        input_file = Path('tests/fixtures/deft_format_samples.deft')
        examples = deft_to_jsonl_converter._convert_deft_file(input_file)
        assert len(examples) == 2
        assert len(examples[0]['sentences']) == 3
        assert len(examples[1]['sentences']) == 3
