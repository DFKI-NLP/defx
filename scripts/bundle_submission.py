import argparse
import csv
import itertools
from pathlib import Path
import sys
from typing import Dict, List, Any
from zipfile import ZipFile

from allennlp.predictors import Predictor
from allennlp.data.instance import Instance
from tqdm import tqdm

sys.path.append('.')
import defx


def batched_predict_instances(predictor: Predictor,
                              examples: List[Instance],
                              batch_size: int = 16) -> List[Dict[str, Any]]:
    results = []  # type: List[Dict[str, Any]]
    for i in tqdm(range(0, len(examples), batch_size)):
        batch_examples = examples[i: i + batch_size]
        batch_results = predictor.predict_batch_instance(batch_examples)
        results.extend(batch_results)
    return results


def write_subtask1_output_file(output_file, predictions):
    with output_file.open('w') as f:
        writer = csv.writer(f, delimiter='\t', quotechar='"',
                            quoting=csv.QUOTE_ALL)
        for prediction in predictions:
            input_file, input_row, text, instance, result = prediction
            expected_origin = f'{input_file}##{input_row}'
            label = 1 if result.get('label') == 'HasDef' else 0
            assert instance.get('origin').metadata == expected_origin
            writer.writerow([text, label])


def write_task1_predictions(input_files, output_dir, predicted_instances):
    sentences = []
    source_files = []
    source_rows = []
    for input_file in input_files:
        sents, files, rows = read_task1_input_file(input_file)
        sentences.extend(sents)
        source_files.extend(files)
        source_rows.extend(rows)

    assert len(sentences) == len(predicted_instances)
    instances, results = zip(*predicted_instances)
    predictions = zip(source_files, source_rows, sentences, instances, results)
    predictions_by_origin = itertools.groupby(predictions,
                                              lambda x: x[0])
    for input_file, prediction_group in predictions_by_origin:
        output_file_name = f'task_1_{input_file}'
        output_file = Path(output_dir, output_file_name)
        write_subtask1_output_file(output_file, prediction_group)


def read_task1_input_file(input_file):
    sentences = []
    source_files = []
    source_rows = []
    with input_file.open() as f:
        for idx, row in enumerate(csv.reader(f, delimiter='\t', quotechar='"')):
            source_files.append(input_file.name)
            source_rows.append(idx)
            sentences.append(row[0])
    return sentences, source_files, source_rows


def write_task2_predictions(output_dir, predicted_instances):
    get_file_name = lambda p: get_instance_source_file(p[0])
    predictions_by_file = itertools.groupby(predicted_instances,
                                            key=get_file_name)
    for input_file, prediction_group in predictions_by_file:
        output_file_name = f'task_2_{input_file}'
        output_file = Path(output_dir, output_file_name)
        write_subtask2_output_file(output_file, list(prediction_group))


def write_subtask2_output_file(output_file, predictions):
    with output_file.open('w') as f:
        writer = csv.writer(f, delimiter='\t', quotechar='"',
                            quoting=csv.QUOTE_ALL)
        for prediction in predictions:
            instance, results = prediction
            source_file = get_instance_source_file(instance)
            for word, tag in zip(results['words'], results['tags']):
                writer.writerow([word, source_file, 666, 666, tag])
            writer.writerow([])


def write_task3_predictions(output_dir, predicted_instances):
    get_file_name = lambda p: get_instance_source_file(p[0])
    predictions_by_file = itertools.groupby(predicted_instances,
                                            key=get_file_name)
    for input_file, prediction_group in predictions_by_file:
        output_file_name = f'task_3_{input_file}'
        output_file = Path(output_dir, output_file_name)
        write_subtask3_output_file(output_file, list(prediction_group))


def write_subtask3_output_file(output_file, predictions):
    with output_file.open('w') as f:
        writer = csv.writer(f, delimiter='\t', quotechar='"',
                            quoting=csv.QUOTE_ALL)
        for prediction in predictions:
            instance, results = prediction
            source_file = get_instance_source_file(instance)
            words = results['words']
            if 'ner_ids' in results:
                ner_ids = results['ner_ids']
            else:
                ner_ids = ['-1'] * len(words)
            if 'predicted_heads' in results:
                predicted_heads = results['predicted_heads']
            else:
                predicted_heads = results['predicted_head_offsets']

            fields = zip(words,
                         results['tags'],
                         ner_ids,
                         predicted_heads,
                         results['predicted_relations'])
            for word, tag, ner_id, head, relation in fields:
                writer.writerow([word, source_file, 666, 666, tag, ner_id, head, relation])
            writer.writerow([])


def get_instance_source_file(instance):
    return instance.get('metadata')['example_id'].split('##')[0]


parser = argparse.ArgumentParser('Bundle predictions for a submission')
parser.add_argument('input_data',
                    help='Folder with .deft files containing input sentences')
parser.add_argument('prediction_output',
                    help='Output folder with files containing predicted labels')
parser.add_argument('--model-archive', help='The model.tar.gz file')
parser.add_argument('--predictor', default='subtask1_predictor')
parser.add_argument('--submission-file', default='submission.zip')
parser.add_argument('--subtasks', nargs="+", type=int, default=[1],
                    help='Choose which subtasks to bundle')
parser.add_argument('-f', dest='force_output', action='store_true',
                    help='force creation of a new output dir')
args = parser.parse_args()

for subtask in args.subtasks:
    assert subtask in [1, 2, 3], 'Subtask not supported: {}'.format(subtask)
input_data = Path(args.input_data)
output_dir = Path(args.prediction_output)
assert input_data.exists()
if output_dir.exists():
    assert args.force_output, 'Output directory already exists'
else:
    output_dir.mkdir(exist_ok=True)

print('Reading instances...')
predictor = Predictor.from_path(args.model_archive, args.predictor)
instances = predictor._dataset_reader.read(args.input_data)

print('Running prediction...')
results = batched_predict_instances(predictor, instances, batch_size=32)
predicted_instances = list(zip(instances, results))

print('Writing predictions...')
if 1 in args.subtasks:
    input_files = input_data.glob('*.deft')
    write_task1_predictions(input_files, output_dir, predicted_instances)
if 2 in args.subtasks:
    write_task2_predictions(output_dir, predicted_instances)
if 3 in args.subtasks:
    write_task3_predictions(output_dir, predicted_instances)

print('Creating archive...')
with ZipFile(Path(output_dir, args.submission_file), 'w') as zf:
    for pred_file in Path(output_dir).iterdir():
        if pred_file.suffix == '.deft':
            zf.write(pred_file, pred_file.name)

print('Done.')
