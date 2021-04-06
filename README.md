# Joint Extraction of Concepts and Relations for Definition Extraction
Source code for the submission on the shared task of Semeval 2020 Task 8 (DeftEval): Extracting term-definition pairs on the DEFT Corpus, an english textbook corpus. The accompanying paper "Defx at SemEval-2020 Task 6: Joint Extraction of Concepts and Relations for Definition Extraction" by Marc Hübner, Christoph Alt, Robert Schwarzenberg, Leonhard Hennig can be found here: https://www.aclweb.org/anthology/2020.semeval-1.92/

### Prerequisites
Tested with **Python 3.7.4**:
```
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### Preprocessing the data

Clone the deft corpus:
```
git clone https://github.com/adobe-research/deft_corpus data/deft_corpus
```

Run the preprocessing script:
```
CUDA_VISIBLE_DEVICES=0 bash scripts/preprocess.sh
```

### Training a new model

Set the random seed (ALLENNLP_SEED) and the cuda device (ALLENNLP_DEVICE) for training:
```
CUDA_VISIBLE_DEVICES=0 \
ALLENNLP_SEED=0 \
ALLENNLP_DEVICE=0 \
allennlp train configs/joint_bert_classifier.jsonnet \
  --include-package defx \
  -s data/runs/joint_model
```

### Evaluating a trained model
```
python scripts/evaluate.py \
  --cuda-device 0 \
  --subtasks 2 \
  --split dev \
  data/runs/joint_model
```

### Running a demo

```
pip install streamlit
streamlit run streamlit_demo.py -- data/runs/joint_model/model.tar.gz
```
