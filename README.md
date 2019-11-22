# Semeval 2020 Task 6
DeftEval: Extracting term-definition pairs in free text

## Prerequisites

### Package requirements
Tested with **Python 3.7.4**:
```
pip install -r requirements.txt
```

### Clone the deft corpus
```
git clone https://github.com/adobe-research/deft_corpus data/deft_corpus
```

### Run preprocessing
```
bash scripts/preprocess.sh
```

## Training a new model

### Run the training
```
ALLENNLP_SEED=0 ALLENNLP_DEVICE=-1 allennlp train configs/subtask1_basic_debug.jsonnet --include-package defx -s data/runs/subtask1_example
```

### Generate predictions
```
python scripts/bundle_submission.py data/deft_split/subtask1_raw/test/ data/results/subtask1_example/ --model-archive data/runs/subtask1_example/model.tar.gz
```

### Call the official evaluation metrics
```
python scripts/evaluate.py data/deft_split/subtask1_raw/dev/ data/results/subtask1_example/
```

## Running a demo

Install streamlit and spacy, and execute:
```
streamlit run streamlit_demo.py -- data/runs/joint_bert_classifier/model.tar.gz
```
