#!/bin/bash

echo "Running data preprocessing..."

echo "Creating data folders..."
mkdir -p data/deft_split/raw/train_dev/
mkdir -p data/deft_split/raw/train/
mkdir -p data/deft_split/raw/dev/
mkdir -p data/deft_split/raw/test/
mkdir -p data/deft_split/jsonl/
mkdir -p data/deft_split/subtask1_raw/train_dev/
mkdir -p data/deft_split/subtask1_raw/train/
mkdir -p data/deft_split/subtask1_raw/dev/
mkdir -p data/deft_split/subtask1_raw/test/

echo "Splitting raw data..."
cp data/deft_corpus/data/deft_files/dev/*.deft data/deft_split/raw/test/
cp data/deft_corpus/data/deft_files/train/*.deft data/deft_split/raw/train_dev/
cp data/deft_corpus/data/deft_files/train/*.deft data/deft_split/raw/train/
# Previously: Move n random files from train to dev
#ls data/deft_split/raw/train/*.deft | shuf -n 8 | while read fn; do mv $fn data/deft_split/raw/dev/; done
# Now: Move the files of the predefined dev set
for fn in `cat scripts/dev_split_files.txt`; do mv data/deft_split/raw/train/$fn data/deft_split/raw/dev/; done

echo "Converting to subtask 1 format..."
python data/deft_corpus/task1_converter.py data/deft_split/raw/train_dev/ data/deft_split/subtask1_raw/train_dev/
python data/deft_corpus/task1_converter.py data/deft_split/raw/train/ data/deft_split/subtask1_raw/train/
python data/deft_corpus/task1_converter.py data/deft_split/raw/dev/ data/deft_split/subtask1_raw/dev/
python data/deft_corpus/task1_converter.py data/deft_split/raw/test/ data/deft_split/subtask1_raw/test/

echo "Converting to jsonl format..."
python defx/util/deft_to_jsonl_converter.py data/deft_split/raw/train/ data/deft_split/jsonl/train.jsonl
python defx/util/deft_to_jsonl_converter.py data/deft_split/raw/dev/ data/deft_split/jsonl/dev.jsonl
python defx/util/deft_to_jsonl_converter.py data/deft_split/raw/test/ data/deft_split/jsonl/test.jsonl
python defx/util/deft_to_jsonl_converter.py data/deft_split/raw/test/ data/deft_split/jsonl/test.jsonl
python defx/util/deft_to_jsonl_converter.py data/deft_corpus/data/test_files/subtask_2 data/deft_split/jsonl/subtask_2.jsonl

echo "Done."
