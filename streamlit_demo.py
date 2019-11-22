import sys

import streamlit as st
from allennlp.predictors import Predictor
from spacy import displacy

import defx
from defx.util.displacy_formatter import DisplacyFormatter

model_name = sys.argv[1] or 'data/runs/joint_bert_classifier/model.tar.gz'
print('Loading model from', model_name)


def setup_predictor():
    pred = Predictor.from_path(model_name, 'joint-classifier')
    return pred


predictor = st.cache(
    setup_predictor,
    allow_output_mutation=True  # the Predictor is not hashable
)()

HTML_WRAPPER = """
<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>
"""
st.title('Defx demo')
text = st.text_input("Model input", "A neural network is a network composed of artificial neurons or nodes")

result = predictor.predict_json({'sentence': text})
displacy_format = DisplacyFormatter().format(result)
html = displacy.render(displacy_format,
                       style="dep",
                       manual=True,
                       options={'compact': True})

# Double newlines seem to mess with the rendering
html = html.replace("\n\n", "\n")
st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
