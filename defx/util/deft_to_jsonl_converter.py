import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, TextIO

from tqdm import tqdm

SENTS_PER_EXAMPLE = 3


def main():
    """Script to convert a folder of deft formatted files into a jsonl file"""
    parser = argparse.ArgumentParser('Convert deft format to jsonl')
    parser.add_argument('input', help='Folder with deft files')
    parser.add_argument('output', help='Jsonl output file')
    parser.add_argument('-f', dest='force_output', action='store_true',
                        help='force creation of a new output file')
    args = parser.parse_args()

    output_file = Path(args.output)
    if output_file.exists():
        assert not output_file.is_dir(), 'Output must be a file'
        assert args.force_output, 'Output file already exists'

    input_path = Path(args.input)
    assert input_path.exists() and input_path.is_dir()

    with output_file.open('w') as output_file_handler:
        _convert_deft_folder(input_path, output_file_handler)


def _convert_deft_folder(input_path: Path, output_file: TextIO) -> None:
    """Convert all files in the given folder."""
    for input_file in tqdm(input_path.iterdir()):
        examples = _convert_deft_file(input_file)
        for example in examples:
            output_file.write(json.dumps(example) + '\n')


def _convert_deft_file(input_file: Path) -> List[Dict[str, Any]]:
    """Converts a deft file into jsonl format and writes to the output file"""
    examples = []
    sentence_count = 0
    with input_file.open() as file_handler:
        while True:
            next_line = _peek_line(file_handler)
            if '\n' not in next_line:
                break

            sentences = _parse_sentence_triple(file_handler)
            for sentence in sentences:
                sentence['id'] = f'{input_file.name}##{sentence_count}'
                sentence_count += 1

            num_sentences = len(sentences)
            sentence_count += num_sentences
            assert 0 < num_sentences < 4, f'invalid sent len: {num_sentences}'

            examples.append({
                'id': f'{input_file.name}##{sentence_count}',
                'sentences': sentences,
                'entities': _extract_entities(sentences),
                'relations': _extract_relations(sentences)
            })
    return examples


def _parse_sentence_triple(file_handler: TextIO) -> List[Dict[str, Any]]:
    """Parses up to three sentences from the given file handler"""
    sentences = []
    for sentence_count in range(SENTS_PER_EXAMPLE):
        sentence = _parse_sentence(file_handler)
        assert len(sentence['tokens']) > 0
        sentences.append(sentence)

        next_line = _peek_line(file_handler)
        if next_line.strip() == '' and sentence_count < 2:
            break  # Break for incomplete tripes

    # Read the empty separator line if present
    next_line = _peek_line(file_handler)
    if next_line.strip() == '':
        file_handler.readline()

    return sentences


def _peek_line(file_handler) -> str:
    """Peeks into the file returns the next line"""
    current_pos = file_handler.tell()
    line = file_handler.readline()
    file_handler.seek(current_pos)
    return line


def _parse_sentence(input_file: TextIO) -> Dict[str, Any]:
    """Parses all lines of the current sentence into a deft sentence object"""
    sentence = {
        'sentence_label': 'NoDef',
        'tokens': [],
        'start_chars': [],
        'end_chars': [],
        'tags': [],
        'ner_ids': [],
        'relation_roots': [],
        'relations': []
    }
    line = input_file.readline()
    while True:
        if line.strip() == '':
            if line == '':
                break  # End of file, stop parsing.
            if _is_chapter_start(sentence):
                line = input_file.readline()
                continue  # End of a chapter, remove the newline and continue
            break  # End of sentence, stop parsing.

        split_line = [l.strip() for l in line.strip().split('\t')]
        assert len(split_line) == 8, 'invalid line format: {}'.format(line)
        sentence['tokens'].append(split_line[0])
        sentence['start_chars'].append(split_line[2])
        sentence['end_chars'].append(split_line[3])
        tag = split_line[4]
        sentence['tags'].append(tag)
        if tag[2:] == 'Definition':
            sentence['sentence_label'] = 'HasDef'
        sentence['ner_ids'].append(split_line[5])
        sentence['relation_roots'].append(split_line[6])
        sentence['relations'].append(split_line[7])
        line = input_file.readline()

    return sentence


def _extract_entities(sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregates entity information into a separate dictionary."""
    entities = []
    ner_info = [info_tuple
                for s in sentences
                for info_tuple in zip(s['tokens'],
                                      s['tags'],
                                      s['ner_ids'])]
    filtered_tokens = [i for i in enumerate(ner_info) if i[1][1] != 'O']
    for entity_id, entity_group in itertools.groupby(filtered_tokens,
                                                     key=lambda x: x[1][2]):
        token_offsets, ner_tuples = zip(*entity_group)
        _, tags, ner_ids = zip(*ner_tuples)

        assert ner_ids[0] == entity_id, "{} != {}".format(ner_ids[0], entity_id)
        assert ner_ids[0] != '-1'

        # Detect issues in the task2 tags annotations
        tags = list(tags)
        if not tags[0].startswith('B-'):
            print('incorrect starting tag: {} of {} in {}'
                  .format(tags[0], entity_id, sentences[0]['id']))
            tags[0] = 'B-' + tags[0]
        for i in range(1, len(tags)):
            if not tags[i].startswith('I-'):
                print('incorrect intermediate tag: {} of {} in {}'
                      .format(tags[i],
                              entity_id,
                              sentences[0]['id']))
                tags[i] = 'I-' + tags[i]

        for i in range(1, len(tags)):
            assert tags[i].startswith('I-')
        assert tags[0].startswith('B-')

        entity_type = tags[0][2:]
        start_token = token_offsets[0]
        end_token = token_offsets[-1] + 1

        entities.append({
            'id': entity_id,
            'entity_type': entity_type,
            'start_token': start_token,
            'end_token': end_token
        })

    return entities


def _extract_relations(sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregates relation information into a separate dictionary."""
    relations = []
    relation_info = [info_tuple
                     for s in sentences
                     for info_tuple in zip(s['ner_ids'],
                                           s['relation_roots'],
                                           s['relations'])]
    relation_tokens = [i for i in relation_info if i[1] not in ['-1', '0']]
    grouped_relations = itertools.groupby(relation_tokens, key=lambda x: x[0])
    for _, relation_group in grouped_relations:
        tail_id, head_id, relation_type = next(relation_group)
        relations.append({
            'head_id': head_id,
            'tail_id': tail_id,
            'relation_type': relation_type
        })

    return relations


def _is_chapter_start(sentence: Dict[str, Any]):
    """
    Return true if the sentence only contains a chapter start.

    Most examples start with a chapter start, i.e. a digit followed by period
    character. This is not supposed to be handled as a separate sentence, but
    the sentence splitting seems to have introduced newlines in these cases.
    """
    if len(sentence['tokens']) != 2:
        return False
    return sentence['tokens'][0].isdigit() and sentence['tokens'][1] == '.'


if __name__ == '__main__':
    main()