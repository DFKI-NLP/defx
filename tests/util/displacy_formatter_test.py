from unittest import TestCase

from defx.util.displacy_formatter import DisplacyFormatter


class DisplacyFormatterTest(TestCase):
    @staticmethod
    def test_complete_basic_formatting():
        formatter = DisplacyFormatter()
        expected_output = {
            "words": [
                {"text": "This", "tag": "DT"},
                {"text": "is", "tag": "VBZ"},
                {"text": "a", "tag": "DT"},
                {"text": "sentence", "tag": "NN"}
            ],
            "arcs": [
                {"start": 0, "end": 1, "label": "nsubj", "dir": "left"},
                {"start": 2, "end": 3, "label": "det", "dir": "left"},
                {"start": 1, "end": 3, "label": "attr", "dir": "right"}
            ]
        }
        prediction = {
            'words': ['This', 'is', 'a', 'sentence'],
            'tags': ['DT', 'VBZ', 'DT', 'NN'],
            'head_offsets': [1, -1, 3, 1],
            'relations': ['nsubj', '0', 'det', 'attr'],
        }
        formatted_output = formatter.format(prediction)
        assert formatted_output['words'] == expected_output['words']
        assert formatted_output == expected_output
