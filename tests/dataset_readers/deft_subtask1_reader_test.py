from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from defx.dataset_readers import DeftSubtask1Reader


class DeftSubtask1ReaderTest(AllenNlpTestCase):
    """Test the implementation of the dataset reader for subtask 1"""

    @staticmethod
    def check_instances(instances):
        """
        Checks that the passed instances have been read correctly.

        Parameters
        ----------
        first : List[Instance]
            A list of instances returned by the reader.
        """
        get_tokens = lambda i: [t.text for t in i.fields['tokens'].tokens]

        first_instance = {
            'tokens': [
                "5", ".", "Science", "includes", "such", "diverse", "fields",
                "as", "astronomy", ",", "biology", ",", "computer", "sciences",
                ",", "geology", ",", "logic", ",", "physics", ",", "chemistry",
                ",", "and", "mathematics", "(", "[", "link", "]", ")", "."
            ],
            'label': 'NoDef'
        }
        assert get_tokens(instances[0]) == first_instance["tokens"]
        assert instances[0]["label"].label == first_instance["label"]

        second_instance = {
            'tokens': [
                "However", ",", "those", "fields", "of", "science", "related",
                "to", "the", "physical", "world", "and", "its", "phenomena",
                "and", "processes", "are", "considered", "natural", "sciences",
                "."
            ],
            'label': 'HasDef'
        }
        assert get_tokens(instances[1]) == second_instance["tokens"]
        assert instances[1]["label"].label == second_instance["label"]

    @staticmethod
    def test_read_single_file():
        """Test that instances are read correctly from a single file"""
        reader = DeftSubtask1Reader()
        instances = ensure_list(
            reader.read('tests/fixtures/deft_subtask1_sample.deft'))

        assert len(instances) == 5
        DeftSubtask1ReaderTest.check_instances(instances)


    @staticmethod
    def test_read_directory():
        """Test that all instances are read correctly from a directory"""
        reader = DeftSubtask1Reader()
        instances = ensure_list(reader.read('tests/fixtures/subtask1_dir'))

        assert len(instances) == 10
        DeftSubtask1ReaderTest.check_instances(instances)
