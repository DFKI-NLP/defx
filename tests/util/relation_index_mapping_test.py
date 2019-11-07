from unittest import TestCase

from defx.util.index_to_relation_and_type_mapping import map_relation_head_and_type_to_index, \
    map_index_to_relation_head_and_type


class RelationIndexMappingTest(TestCase):
    """
    Tests mapping between relation scorer indexes, and relation head and type
    """

    @staticmethod
    def test_map_index_to_relation_head_and_type():
        # Example for 3 classes and seq_len = 3:
        # idx = 0 -> head_idx = -1, type = 'negative_label'
        # idx = 1 -> head_idx =  0, type = 'positive_label_1'
        # idx = 2 -> head_idx =  0, type = 'positive_label_2'
        # idx = 3 -> head_idx =  1, type = 'positive_label_1'
        # idx = 4 -> head_idx =  1, type = 'positive_label_2'
        # idx = 5 -> head_idx =  2, type = 'positive_label_1'
        # idx = 6 -> head_idx =  2, type = 'positive_label_2'
        label_vocab = {
            0: 'negative_label',
            1: 'positive_label_1',
            2: 'positive_label_2'
        }

        def map_idx(idx: int):
            return map_index_to_relation_head_and_type(
                label_vocab=label_vocab, head_and_type_idx=idx
            )

        assert map_idx(0) == (-1, 'negative_label')
        assert map_idx(1) == (0, 'positive_label_1')
        assert map_idx(2) == (0, 'positive_label_2')
        assert map_idx(3) == (1, 'positive_label_1')
        assert map_idx(4) == (1, 'positive_label_2')
        assert map_idx(5) == (2, 'positive_label_1')
        assert map_idx(6) == (2, 'positive_label_2')

    @staticmethod
    def test_map_relation_head_and_type_to_index():
        # Example for 3 classes and seq_len = 3:
        #               class_idx = 0 -> idx = 0
        # head_idx = 0, class_idx = 1 -> idx = 1
        # head_idx = 0, class_idx = 2 -> idx = 2
        # head_idx = 1, class_idx = 1 -> idx = 3
        # head_idx = 1, class_idx = 2 -> idx = 4
        # head_idx = 2, class_idx = 1 -> idx = 5
        # head_idx = 2, class_idx = 2 -> idx = 6

        def map_head_and_type(rel_head: int, rel_type: int):
            return map_relation_head_and_type_to_index(
                num_pos_types=2,
                relation_type=rel_type,
                relation_head=rel_head
            )

        # For the negative class all heads map to idx 0
        assert map_head_and_type(0, 0) == 0
        assert map_head_and_type(1, 0) == 0
        assert map_head_and_type(2, 0) == 0

        assert map_head_and_type(0, 1) == 1
        assert map_head_and_type(0, 2) == 2
        assert map_head_and_type(1, 1) == 3
        assert map_head_and_type(1, 2) == 4
        assert map_head_and_type(2, 1) == 5
        assert map_head_and_type(2, 2) == 6
