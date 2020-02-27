import csv
import sys
import warnings
from pprint import pprint
from statistics import mean, stdev

# Import the official evaluation scripts
sys.path.append('data/deft_corpus/evaluation/program')
from evaluation_sub1 import get_gold_and_pred_labels as task1_get_labels
from evaluation_sub1 import evaluate as task1_evaluate
from evaluation_sub2 import evaluate as task2_evaluate
from evaluation_sub2 import validate_data, get_label
from evaluation_sub3 import evaluate as task3_evaluate
from evaluation_sub3 import has_relation, get_relation, get_relation_from, get_relation_to


def evaluate_subtasks(subtasks, gold_dir, pred_dir, eval_config, verbose=False, quiet=False):
    result = {}
    assert not (quiet and verbose)
    for subtask in subtasks:
        eval_labels = eval_config[f'task_{subtask}']['eval_labels']
        task_report = _evaluate_subtask(subtask=subtask,
                                        eval_labels=eval_labels,
                                        gold_dir=gold_dir,
                                        pred_dir=pred_dir)
        if verbose:
            print(f'Subtask {subtask} report:')
            pprint(task_report)
        f1_score = _get_f1_score(task_report, subtask=subtask)
        if not quiet:
            print(f'Subtask {subtask} score: {f1_score * 100:.2f} F1')
        result[f'subtask{subtask}'] = task_report
    return result


def aggregate_results(results):
    summary = {}
    subtasks = [int(task_label[len('subtask'):])
                for task_label in results[0].keys()]
    for subtask in subtasks:
        task_label = f'subtask{subtask}'

        precision_values = [_get_precision(r[task_label], subtask) for r in results]
        p_mean = mean(precision_values)
        p_std = stdev(precision_values)
        summary[f'{task_label}_precision_mean'] = p_mean
        summary[f'{task_label}_precision_stdev'] = p_std

        recall_values = [_get_recall(r[task_label], subtask) for r in results]
        r_mean = mean(recall_values)
        r_std = stdev(recall_values)
        summary[f'{task_label}_recall_mean'] = r_mean
        summary[f'{task_label}_recall_stdev'] = r_std

        f1_values = [_get_f1_score(r[task_label], subtask) for r in results]
        f1_mean = mean(f1_values)
        f1_std = stdev(f1_values)

        summary[f'{task_label}_f1_mean'] = f1_mean
        summary[f'{task_label}_f1_stdev'] = f1_std

        print(f'Subtask {subtask} score:')
        print(f'  P: {p_mean * 100:.2f}+-{p_std * 100:.2f}')
        print(f'  R: {r_mean * 100:.2f}+-{r_std * 100:.2f}')
        print(f'  F: {f1_mean * 100:.2f}+-{f1_std * 100:.2f}')

    return summary


def _get_precision(result, subtask):
    if subtask == 1:
        return result["1"]["precision"]
    elif subtask == 2:
        return result["macro avg"]["precision"]
    elif subtask == 3:
        return result["macro"]["p"]
    else:
        raise RuntimeError('Unknown subtask')


def _get_recall(result, subtask):
    if subtask == 1:
        return result["1"]["recall"]
    elif subtask == 2:
        return result["macro avg"]["recall"]
    elif subtask == 3:
        return result["macro"]["r"]
    else:
        raise RuntimeError('Unknown subtask')


def _get_f1_score(result, subtask):
    if subtask == 1:
        return result["1"]["f1"]
    elif subtask == 2:
        return result["macro avg"]["f1-score"]
    elif subtask == 3:
        return result["macro"]["f"]
    else:
        raise RuntimeError('Unknown subtask')


def _task2_get_labels(gold_fname, pred_fname):
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
        pred_reader = csv.reader(pred_source, delimiter="\t", quoting=csv.QUOTE_NONE)
        pred_rows = [row for row in pred_reader if row]

    validate_data(gold_rows, pred_rows)
    y_gold = [get_label(row) for row in gold_rows]
    y_pred = [get_label(row) for row in pred_rows]
    return y_gold, y_pred


def _task3_get_labels(gold_fname, pred_fname):
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

    gold_relation_rows = [row for row in gold_rows if _has_relation_and_head(row)]
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


def _evaluate_subtask(subtask, eval_labels, gold_dir, pred_dir):
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
                file_y_gold, file_y_pred = _task2_get_labels(gold_file, pred_file)
            elif subtask == 3:
                file_y_gold, file_y_pred = _task3_get_labels(gold_file, pred_file)
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


def _has_relation_and_head(row):
    """Does this token participate in a relation?"""
    return row[-1] != "0" and row[-2] != "0"
