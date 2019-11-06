import argparse
import csv
import warnings
from pathlib import Path
from pprint import pprint
import sys

# Import the official evaluation scripts
from yaml import safe_load

sys.path.append('data/deft_corpus/evaluation/program')
from evaluation_sub1 import get_gold_and_pred_labels as task1_get_labels
from evaluation_sub1 import evaluate as task1_evaluate
from evaluation_sub2 import evaluate as task2_evaluate
from evaluation_sub2 import validate_data, get_label


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
            else:
                raise RuntimeError('Unknown subtask')
            y_gold.extend(file_y_gold)
            y_pred.extend(file_y_pred)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore scikit learn warnings
        if subtask == 1:
            report = task1_evaluate(y_gold, y_pred, eval_labels)
        elif subtask == 2:
            report = task2_evaluate(y_gold, y_pred, eval_labels)
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
        gold_reader = csv.reader(gold_source, delimiter="\t", quoting=csv.QUOTE_NONE)
        gold_rows = [[col.strip() for col in row[:5]] for row in gold_reader if row]

    with pred_fname.open() as pred_source:
        pred_reader = csv.reader(pred_source, delimiter="\t")
        pred_rows = [row for row in pred_reader if row]

    validate_data(gold_rows, pred_rows)
    y_gold = [get_label(row) for row in gold_rows]
    y_pred = [get_label(row) for row in pred_rows]
    return y_gold, y_pred


parser = argparse.ArgumentParser(description='Runs the official evaluation')
parser.add_argument('gold_input', type=str,
                    help='Gold labeled documents')
parser.add_argument('pred_input', type=str,
                    help='Predicted document labels')
parser.add_argument('--verbose', action='store_true',
                    help='Print verbose metrics')
parser.add_argument('--subtasks', nargs="+", type=int,
                    default=[1], help='Choose which subtasks to evaluate')
args = parser.parse_args()

for subtask in args.subtasks:
    assert subtask in [1, 2], 'Subtask not supported: {}'.format(subtask)

gold_dir = Path(args.gold_input)
pred_dir = Path(args.pred_input)
config_path = Path('data/deft_corpus/evaluation/program/configs/eval_test.yaml')
assert config_path.exists() and config_path.is_file(), 'Config not found'
with config_path.open() as cfg_file:
    config = safe_load(cfg_file)

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
