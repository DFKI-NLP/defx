from typing import Dict


def map_index_to_relation_head_and_type(label_vocab: Dict[int, str],
                                        head_and_type_idx: int) -> (int, str):
    if head_and_type_idx == 0:
        # The negative class (assumed to be at index 0) is always pointing to -1
        return -1, label_vocab[0]
    else:
        head_and_type_idx -= 1
        num_pos_classes = len(label_vocab) - 1
        relation_head = head_and_type_idx // num_pos_classes
        relation_type_idx = head_and_type_idx % num_pos_classes + 1
        relation_type = label_vocab[relation_type_idx]
        return relation_head, relation_type


def map_relation_head_and_type_to_index(num_pos_types: int,
                                        relation_head: int,
                                        relation_type: int) -> int:
    if relation_type == 0:  # negative class assumed at index 0
        return 0
    else:
        return relation_head * num_pos_types + relation_type

