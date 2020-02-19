# Semeval 2020 Task 6
DeftEval: Extracting term-definition pairs in free text

## Prerequisites

### Package requirements
Tested with **Python 3.7.4**:
```
pip install -r requirements.txt
python -m spacy download en_core_web_lg
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

### Evaluate the model and generate a submission
```
python scripts/evaluate.py --cuda-device <device-id> <model-dir> --subtasks <1-3> --split <dev/test>
```

## Running a demo

```
pip install streamlit
streamlit run streamlit_demo.py -- data/runs/joint_bert_classifier/model.tar.gz
```
