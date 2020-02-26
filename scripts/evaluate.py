import argparse
import json
import os
from pathlib import Path
import sys

from yaml import safe_load

sys.path.append('.')
from defx.util.official_evaluation import evaluate_subtasks, aggregate_results
from defx.util.predictions_writer import PredictionsWriter


def main():
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
    parser.add_argument('--only-eval', dest='only_eval', action='store_true',
                        help='Only evaluate a given folder of predictions')
    parser.add_argument('--cuda-device', default=-1, type=int,
                        help='The cuda device used for predictions')
    parser.add_argument('-f', dest='force_pred', action='store_true',
                        help='Force creation of new predictions')
    args = parser.parse_args()

    for subtask in args.subtasks:
        assert subtask in [1, 2, 3], 'Subtask not supported: {}'.format(subtask)

    raw_base_path = 'data/deft_split/raw/'
    jsonl_base_path = 'data/deft_split/jsonl/'
    split = args.split
    model_input = Path(jsonl_base_path, f'{split}.jsonl')
    gold_dir = Path(raw_base_path, split)

    config_path = Path('data/deft_corpus/evaluation/program/configs/eval_test.yaml')
    assert config_path.exists() and config_path.is_file(), 'Config not found'
    with config_path.open() as cfg_file:
        config = safe_load(cfg_file)

    results = []
    if args.only_eval:
        cwd = pred_dir = args.model_dir
        result = evaluate_subtasks(args.subtasks,
                                   gold_dir=gold_dir,
                                   pred_dir=Path(pred_dir),
                                   eval_config=config,
                                   verbose=args.verbose)

        with Path(cwd, f'{split}_results.json').open('w') as f:
            json.dump(result, f)
        results.append(result)
    else:
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

                if _is_challenge_dataset(split):  # Skip evaluation for submissions
                    print('Skipping evaluation for challenge test set')
                    continue

                result = evaluate_subtasks(args.subtasks,
                                           gold_dir=gold_dir,
                                           pred_dir=pred_dir,
                                           eval_config=config,
                                           verbose=args.verbose)

                with Path(cwd, f'{split}_results.json').open('w') as f:
                    json.dump(result, f)
                results.append(result)

    assert len(results) > 0 or _is_challenge_dataset(split), 'Missing results'
    if len(results) > 1:
        # Compute summaries over multiple models
        print('Model summary:')
        summary = aggregate_results(results)
        with Path(args.model_dir, f'{split}_summary.json').open('w') as f:
            json.dump(summary, f)


def _is_challenge_dataset(split: str):
    return split.startswith('subtask')


if __name__ == '__main__':
    main()
