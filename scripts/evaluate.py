import argparse
from pathlib import Path
import sys

# Import the official evaluation scripts
sys.path.append('data/deft_corpus/evaluation')
from semeval2020_0601_eval import task_1_eval_main


parser = argparse.ArgumentParser(description='Evaluation script (for subtask 1)')
parser.add_argument('gold_file', type=str,
                    help='Path to file containing gold labels')
parser.add_argument('pred_file', type=str,
                    help='Path to file containing predicted labels')

args = parser.parse_args()
print(args)

gold_file = Path(args.gold_file)
pred_file = Path(args.pred_file)
eval_labels = None  # irrelevant for subtask 1
report = task_1_eval_main(gold_file, pred_file, eval_labels)
print(report)
