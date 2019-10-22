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


def write_predictions(output_file, predictions, string_label=True):
    with output_file.open('w') as f:
        writer = csv.writer(f, delimiter='\t', quotechar='"',
                            quoting=csv.QUOTE_ALL)
        for prediction in predictions:
            input_file, input_row, text, instance, result = prediction
            expected_origin = f'{input_file}##{input_row}'
            if string_label:
                label = result.get('label')
            else:
                label = 1 if result.get('label') == 'HasDef' else 0
            if not args.single_file:
                assert instance.get('origin').metadata == expected_origin
            writer.writerow([text, label])

def read_input_file(input_file):
    sentences = []
    source_files = []
    source_rows = []
    with input_file.open() as f:
        for idx, row in enumerate(csv.reader(f, delimiter='\t', quotechar='"')):
            source_files.append(input_file.name)
            source_rows.append(idx)
            sentences.append(row[0])
    return sentences, source_files, source_rows


parser = argparse.ArgumentParser('Bundle predictions for a submission')
parser.add_argument('input_data',
                    help='Folder with .deft files containing input sentences')
parser.add_argument('prediction_output',
                    help='Output folder with files containing predicted labels')
parser.add_argument('--model-archive', help='The model.tar.gz file')
parser.add_argument('--single-file', action='store_true',
                    help='Skip the archive creation and generate a single file')
args = parser.parse_args()

print('Reading instances...')
predictor = Predictor.from_path(args.model_archive, 'subtask1_predictor')
instances = predictor._dataset_reader.read(args.input_data)

print('Running prediction...')
results = batched_predict_instances(predictor, instances, batch_size=32)

print('Reading source texts...')
input_data = Path(args.input_data)
if input_data.is_dir():
    sentences = []
    source_files = []
    source_rows = []
    for input_file in Path(args.input_data).iterdir():
        sents, files, rows = read_input_file(input_file)
        sentences.extend(sents)
        source_files.extend(files)
        source_rows.extend(rows)
else:
    sentences, source_files, source_rows = read_input_file(Path(input_data))


assert len(sentences) == len(results)
predictions = zip(source_files, source_rows, sentences, instances, results)

print('Writing predictions...')
if args.single_file:
    output_file = Path(args.prediction_output)
    assert not output_file.exists() or output_file.is_file()
    write_predictions(output_file, predictions, string_label=False)
else:
    output_dir = Path(args.prediction_output)
    assert not output_dir.exists() or output_dir.is_dir()
    output_dir.mkdir(exist_ok=True)
    predictions_by_origin = itertools.groupby(predictions,
                                              lambda x: x[0])
    for input_file, prediction_group in predictions_by_origin:
        output_file_name = 'task_1_' + input_file
        output_file = Path(output_dir, output_file_name)
        write_predictions(output_file, prediction_group)

if not args.single_file:
    print('Creating archive...')
    with ZipFile(Path(output_dir, "task_1_submission.zip"), 'w') as zf:
        for pred_file in Path(output_dir).iterdir():
            if pred_file.suffix == '.deft':
                zf.write(pred_file, pred_file.name)
                pred_file.unlink()

print('Done.')
