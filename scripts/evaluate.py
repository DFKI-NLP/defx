import argparse
import csv
import json
import os
import warnings
from pathlib import Path
from pprint import pprint
from statistics import mean, stdev
import sys

# Import the official evaluation scripts
from yaml import safe_load

sys.path.append('.')
from defx.util.predictions_writer import PredictionsWriter

sys.path.append('data/deft_corpus/evaluation/program')
from evaluation_sub1 import get_gold_and_pred_labels as task1_get_labels
from evaluation_sub1 import evaluate as task1_evaluate
from evaluation_sub2 import evaluate as task2_evaluate
from evaluation_sub2 import validate_data, get_label
from evaluation_sub2 import validate_columns, validate_length
from evaluation_sub3 import get_gold_and_pred_relations as task3_get_labels
from evaluation_sub3 import evaluate as task3_evaluate
from evaluation_sub3 import has_relation, get_relation, get_relation_from, get_relation_to


def evaluate_subtask(subtask, eval_labels, gold_dir, pred_dir):
    y_gold = []
    y_pred = []

    for gold_file in gold_dir.iterdir():
        if gold_file.suffix == '.deft':
            pred_file = pred_dir.joinpath(f"task_{subtask}_{gold_file.name}")

            if not pred_file.exists() or not pred_file.is_file():
                error_message = "Expected submission file '{0}'"
                sys.exit(error_message.format(pred_file))

            if subtask == 1:
                file_y_gold, file_y_pred = task1_get_labels(gold_file, pred_file)
            elif subtask == 2:
                file_y_gold, file_y_pred = task2_get_labels(gold_file, pred_file)
            elif subtask == 3:
                file_y_gold, file_y_pred = task3_get_labels(gold_file, pred_file)
            else:
                raise RuntimeError('Unknown subtask')
            y_gold.extend(file_y_gold)
            y_pred.extend(file_y_pred)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore scikit learn warnings
        if subtask == 1:
            report = task1_evaluate(y_gold, y_pred, eval_labels)
        elif subtask == 2:
            y_pred = [y_label if y_label in eval_labels else 'O'
                      for y_label in y_pred]
            report = task2_evaluate(y_gold, y_pred, eval_labels)
        elif subtask == 3:
            y_pred = [y_label if y_label in eval_labels else '0'
                      for y_label in y_pred]
            report = task3_evaluate(y_gold, y_pred, eval_labels)
        else:
            raise RuntimeError('Unknown subtask')

    return report


def task2_get_labels(gold_fname, pred_fname):
    """Get the labels for evaluation
    Inputs:
        gold_fname: path to .deft file
        pred_fname: path to .deft file
    Returns:
        y_gold: list of labels (strings)
        y_pred: list of labels (strings)
    """
    with gold_fname.open() as gold_source:
        gold_reader = csv.reader(gold_source, delimiter="\t")
        gold_rows = [[col.strip() for col in row[:5]] for row in gold_reader if row]

    with pred_fname.open() as pred_source:
        pred_reader = csv.reader(pred_source, delimiter="\t")
        pred_rows = [row for row in pred_reader if row]

    validate_task2_data(gold_rows, pred_rows)
    y_gold = [get_label(row) for row in gold_rows]
    y_pred = [get_label(row) for row in pred_rows]
    return y_gold, y_pred


def validate_task2_data(gold_rows, pred_rows):
    """Make sure the data is OK
    Inputs:
        gold_rows: list of lists of strings
        pred_rows: list of lists of strings
    """
    validate_length(gold_rows, pred_rows)
    validate_columns(gold_rows, pred_rows)
    # validate_tokens(gold_rows, pred_rows)  # Do not validate the tokens since the gold data is incorrect


def task3_get_labels(gold_fname, pred_fname):
    """Get the relation pairs for evaluation
    Inputs:
        gold_fname: path to .deft file
        pred_fname: path to .deft file
    Returns:
        y_gold_rel_pairs: list of (tag1, tag2, relation) tuples
        y_pred_rel_pairs: list of (tag1, tag2, relation) tuples
    """
    y_gold_rel_pairs = set()  # [(elem1, elem2, rel)]
    y_pred_rel_pairs = set()  # [(elem1, elem2, rel)]

    with gold_fname.open() as gold_source:
        gold_reader = csv.reader(gold_source, delimiter="\t", quoting=csv.QUOTE_NONE)
        gold_rows = [[col.strip() for col in row] for row in gold_reader if row]

    with pred_fname.open() as pred_source:
        pred_reader = csv.reader(pred_source, delimiter="\t")
        pred_rows = [row for row in pred_reader if row]

    validate_data(gold_rows, pred_rows)

    gold_relation_rows = [row for row in gold_rows if has_relation_and_head(row)]
    pred_relation_rows = [row for row in pred_rows if has_relation(row)]

    for row in gold_relation_rows:
        relation = get_relation(row)
        relation_from = get_relation_from(row)
        relation_to = get_relation_to(row)
        y_gold_rel_pairs.add((relation_from, relation_to, relation))

    for row in pred_relation_rows:
        relation = get_relation(row)
        relation_from = get_relation_from(row)
        relation_to = get_relation_to(row)
        y_pred_rel_pairs.add((relation_from, relation_to, relation))
    return y_gold_rel_pairs, y_pred_rel_pairs


def has_relation_and_head(row):
    """Does this token participate in a relation?"""
    return row[-1] != "0" and row[-2] != "0"


parser = argparse.ArgumentParser(
    description='Runs model prediction, bundling, and official evaluation'
)
parser.add_argument('model_dir', type=str,
                    help='Directory with trained model(s)')
parser.add_argument('--predictor', default='subtask1_predictor')
parser.add_argument('--verbose', action='store_true',
                    help='Print verbose metrics')
parser.add_argument('--subtasks', nargs="+", type=int,
                    default=[1], help='Choose which subtasks to evaluate')
parser.add_argument('--split', default='dev',
                    help='The dataset split for evaluation')
parser.add_argument('--cuda-device', default=-1, type=int,
                    help='The cuda device used for predictions')
parser.add_argument('-f', dest='force_pred', action='store_true',
                    help='Force creation of new predictions')
args = parser.parse_args()

for subtask in args.subtasks:
    assert subtask in [1, 2, 3], 'Subtask not supported: {}'.format(subtask)

RAW_BASE_PATH = 'data/deft_split/raw/'
JSONL_BASE_PATH = 'data/deft_split/jsonl/'
split = args.split
model_input = Path(JSONL_BASE_PATH, f'{split}.jsonl')
gold_dir = Path(RAW_BASE_PATH, split)

config_path = Path('data/deft_corpus/evaluation/program/configs/eval_test.yaml')
assert config_path.exists() and config_path.is_file(), 'Config not found'
with config_path.open() as cfg_file:
    config = safe_load(cfg_file)

results = []
for cwd, _, curr_files in os.walk(args.model_dir):
    if 'model.tar.gz' in curr_files:
        pred_dir = Path(cwd, f'{split}_submission')
        model_archive = Path(cwd, 'model.tar.gz')
        print('Evaluating model:', model_archive)
        pred_writer = PredictionsWriter(
            input_data=model_input,
            output_dir=pred_dir,
            model_archive=model_archive,
            predictor=args.predictor,
            subtasks=args.subtasks,
            cuda_device=args.cuda_device,
            task_config=config
        )
        if args.force_pred or pred_writer.missing_predictions():
            pred_writer.run()  # Generate predictions and bundle a submission

        result = {}

        if 1 in args.subtasks:
            eval_labels = config['task_1']['eval_labels']
            task1_report = evaluate_subtask(subtask=1,
                                            eval_labels=eval_labels,
                                            gold_dir=gold_dir,
                                            pred_dir=pred_dir)
            if args.verbose:
                print('Subtask1 report:')
                pprint(task1_report)
            print(f'Subtask 1 score: {task1_report["1"]["f1-score"]*100:.2f} F1')
            result['subtask1'] = task1_report

        if 2 in args.subtasks:
            eval_labels = config['task_2']['eval_labels']
            task2_report = evaluate_subtask(subtask=2,
                                            eval_labels=eval_labels,
                                            gold_dir=gold_dir,
                                            pred_dir=pred_dir)
            if args.verbose:
                print('Subtask2 report:')
                pprint(task2_report)
            print(f'Subtask 2 score: {task2_report["macro avg"]["f1-score"]*100:.2f} F1')
            result['subtask2'] = task2_report

        if 3 in args.subtasks:
            eval_labels = config['task_3']['eval_labels']
            task3_report = evaluate_subtask(subtask=3,
                                            eval_labels=eval_labels,
                                            gold_dir=gold_dir,
                                            pred_dir=pred_dir)
            if args.verbose:
                print('Subtask3 report:')
                pprint(task3_report)
            print(f'Subtask 3 score: {task3_report["macro"]["f"]*100:.2f} F1')
            result['subtask3'] = task3_report

        with Path(cwd, f'{split}_results.json').open('w') as f:
            json.dump(result, f)
        results.append(result)

assert len(results) > 0, 'Missing results'
if len(results) > 1:
    # Compute summaries over multiple models

    print('Model summary:')
    summary = {}
    if 1 in args.subtasks:
        precision_values = [r['subtask1']["1"]["precision"] for r in results]
        summary['subtask1_precision_mean'] = mean(precision_values)
        summary['subtask1_precision_stdev'] = stdev(precision_values)
        recall_values = [r['subtask1']["1"]["recall"] for r in results]
        summary['subtask1_recall_mean'] = mean(recall_values)
        summary['subtask1_recall_stdev'] = stdev(recall_values)
        f1_values = [r['subtask1']["1"]["f1"] for r in results]
        f1_mean = mean(f1_values)
        f1_std = stdev(f1_values)
        summary['subtask1_f1_mean'] = f1_mean
        summary['subtask1_f1_stdev'] = f1_std
        print(f'Subtask 1 score: {f1_mean * 100:.2f}+-{f1_std * 100:.2f} F1')

    if 2 in args.subtasks:
        precision_values = [r['subtask2']["macro avg"]["precision"] for r in results]
        p_mean = mean(precision_values)
        p_std = stdev(precision_values)
        summary['subtask2_precision_mean'] = p_mean
        summary['subtask2_precision_stdev'] = p_std
        recall_values = [r['subtask2']["macro avg"]["recall"] for r in results]
        r_mean = mean(recall_values)
        r_std = stdev(recall_values)
        summary['subtask2_recall_mean'] = r_mean
        summary['subtask2_recall_stdev'] = r_std
        f1_values = [r['subtask2']["macro avg"]["f1-score"] for r in results]
        f1_mean = mean(f1_values)
        f1_std = stdev(f1_values)
        summary['subtask2_f1_mean'] = f1_mean
        summary['subtask2_f1_stdev'] = f1_std
        print(f'Subtask 2 score')
        print(f'  P: {p_mean * 100:.2f}+-{p_std * 100:.2f}')
        print(f'  R: {r_mean * 100:.2f}+-{r_std * 100:.2f}')
        print(f'  F: {f1_mean * 100:.2f}+-{f1_std * 100:.2f}')

    if 3 in args.subtasks:
        precision_values = [r['subtask3']["macro"]["p"] for r in results]
        p_mean = mean(precision_values)
        p_std = stdev(precision_values)
        summary['subtask3_precision_mean'] = p_mean
        summary['subtask3_precision_stdev'] = p_std
        recall_values = [r['subtask3']["macro"]["r"] for r in results]
        r_mean = mean(recall_values)
        r_std = stdev(recall_values)
        summary['subtask3_recall_mean'] = r_mean
        summary['subtask3_recall_stdev'] = r_std
        f1_values = [r['subtask3']["macro"]["f"] for r in results]
        f1_mean = mean(f1_values)
        f1_std = stdev(f1_values)
        summary['subtask3_f1_mean'] = f1_mean
        summary['subtask3_f1_stdev'] = f1_std
        print(f'Subtask 3 score')
        print(f'  P: {p_mean * 100:.2f}+-{p_std * 100:.2f}')
        print(f'  R: {r_mean * 100:.2f}+-{r_std * 100:.2f}')
        print(f'  F: {f1_mean * 100:.2f}+-{f1_std * 100:.2f}')

    with Path(args.model_dir, 'summary.json').open('w') as f:
        json.dump(summary, f)
