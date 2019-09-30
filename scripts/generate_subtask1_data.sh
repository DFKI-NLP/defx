#!/bin/bash

echo "Running subtask 1 data generation script..."

echo "Generate directories for subtask 1 data..."
mkdir -p data/subtask1/raw/train_dev
mkdir -p data/subtask1/raw/test
mkdir -p data/subtask1/split

echo "Running conversion script..."
python data/deft_corpus/task1_converter.py data/deft_corpus/data/deft_files/train/ data/subtask1/raw/train_dev
python data/deft_corpus/task1_converter.py data/deft_corpus/data/deft_files/dev/ data/subtask1/raw/test

echo "Merging files..."
cat data/subtask1/raw/test/* > data/subtask1/split/test.tsv
cat data/subtask1/raw/train_dev/* > data/subtask1/split/train_dev.tsv

echo "Generating train-dev-split..."
python scripts/split_subtask1_train_dev.py

echo "Subtask1 generation finished."
