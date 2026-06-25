"""Dataset generation subpackage for FormInv.

Exposes theorem loaders (:func:`load_curated_theorems`,
:func:`load_from_huggingface`) and paraphrase generation
(:func:`generate_paraphrases`). The :mod:`families` and
:mod:`paraphrases_v2` modules define the 8-family paraphrase taxonomy and its
generator.
"""

from .paraphrases import generate_paraphrases
from .theorems import load_curated_theorems, load_from_huggingface
