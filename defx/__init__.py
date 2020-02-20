"""AllenNLP module for the DeftEval definition extraction task"""

# pylint: disable=wildcard-import
EVALUATED_SUBTASK2_LABELS = [
    'Term', 'Definition', 'Alias-Term', 'Referential-Definition', 'Referential-Term', 'Qualifier'
]
EVALUATED_SUBTASK3_LABELS = [
    'Direct-Defines', 'Indirect-Defines', 'AKA', 'Refers-To', 'Supplements'
]

from defx.dataset_readers import *
from defx.models import *
from defx.modules import *
from defx.predictors import *
from defx.util import *
