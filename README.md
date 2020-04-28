# Joint Extraction of Concepts and Relations for Definition Extraction
Source code for the submission on the shared task of Semeval 2020 Task 8 (DeftEval): Extracting term-definition pairs on the DEFT Corpus, an english textbook corpus.

### Package requirements
Tested with **Python 3.7.4**:
```
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### Preprocessing

Clone the deft corpus
```
git clone https://github.com/adobe-research/deft_corpus data/deft_corpus
```

Run the preprocessing script
```
bash scripts/preprocess.sh
```

### Training a new model

```
ALLENNLP_SEED=0 ALLENNLP_DEVICE=-1 allennlp train configs/subtask1_basic_debug.jsonnet --include-package defx -s data/runs/subtask1_example
```

### Evaluate a model
```
python scripts/evaluate.py --cuda-device <device-id> <model-dir> --subtasks <1-3> --split <dev/test>
```

### Running a demo

```
pip install streamlit
streamlit run streamlit_demo.py -- <model-dir>/model.tar.gz
```
