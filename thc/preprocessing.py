from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from transformers import DistilBertTokenizerFast

from thc import transforms

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


TRANSFORMS: Callable[[str], str] = transforms.Compose(
    transforms=[
        transforms.Demojize(),
        transforms.WordRemove(words_to_remove="@anonymized_account"),
    ]
)
TOKENIZER: PreTrainedTokenizerBase = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-multilingual-cased",
    do_lower_case=False,
    strip_accents=False,
)
