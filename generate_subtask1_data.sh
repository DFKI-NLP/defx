#!/bin/bash

echo "Generate directories for subtask 1 data"
mkdir -p data/subtask1/train
mkdir -p data/subtask1/dev

echo "Generating subtask1 train data"
python data/deft_corpus/task1_converter.py data/deft_corpus/data/deft_files/train/ data/subtask1/train
echo "Generating subtask1 dev data"
python data/deft_corpus/task1_converter.py data/deft_corpus/data/deft_files/dev/ data/subtask1/dev
