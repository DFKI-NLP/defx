#!/bin/bash

echo "Creating folders..."
mkdir -p data/deft_split/raw/train/
mkdir -p data/deft_split/raw/dev/
mkdir -p data/deft_split/raw/test/
mkdir -p data/deft_split/jsonl/

echo "Splitting raw data..."
cp data/deft_corpus/data/deft_files/dev/*.deft data/deft_split/raw/test/
cp data/deft_corpus/data/deft_files/train/*.deft data/deft_split/raw/train/
ls data/deft_split/raw/train/*.deft | shuf -n 8 | while read fn; do mv $fn data/deft_split/raw/dev/; done

echo "Converting to jsonl format..."
python defx/util/deft_to_jsonl_converter.py data/deft_split/raw/train/ data/deft_split/jsonl/train.jsonl
python defx/util/deft_to_jsonl_converter.py data/deft_split/raw/dev/ data/deft_split/jsonl/dev.jsonl
python defx/util/deft_to_jsonl_converter.py data/deft_split/raw/test/ data/deft_split/jsonl/test.jsonl

echo "Done."
